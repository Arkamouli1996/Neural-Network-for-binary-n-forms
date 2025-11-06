"""
TDA Analysis of the Triple-Root Surface in P^4
===============================================
Topological Data Analysis following README_TDA_singular_23_P4.md

This script:
1. Loads positive samples (complex points in C^5 = R^10)
2. Computes global persistent homology (Vietoris-Rips)
3. Detects singular locus via local homology & PCA
4. Creates 3D embeddings and visualizations
5. Generates meshes and exports plots

Usage:
    python run_tda.py --csv ../data/positive_samples.csv --neighbors 50 --outdir ../plot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import argparse
import os
from pathlib import Path

print("Loading libraries...")
try:
    from ripser import ripser
    from persim import plot_diagrams
    print("✓ ripser, persim")
except ImportError:
    print("⚠ ripser/persim not found. Install with: pip install ripser persim")
    ripser = None

try:
    import umap
    print("✓ umap")
except ImportError:
    print("⚠ umap not found. Install with: pip install umap-learn")
    umap = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
    print("✓ plotly")
except ImportError:
    print("⚠ plotly not found. Install with: pip install plotly")
    px = None

try:
    import open3d as o3d
    print("✓ open3d")
except ImportError:
    print("⚠ open3d not found. Install with: pip install open3d")
    o3d = None

try:
    import gudhi as gd
    print("✓ gudhi")
except ImportError:
    print("⚠ gudhi not found. Install with: pip install gudhi")
    gd = None

print()


def normalize_projective_rows(X):
    """Normalize each row to unit norm and apply canonical representative."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    Y = X / norms
    # Canonical representative: force first nonzero entry positive
    for i, row in enumerate(Y):
        j = np.argmax(np.abs(row) > 1e-14)
        if j < row.size and row[j] < 0:
            Y[i] = -row
    return Y


def antipodal_distances(Y):
    """
    Compute antipodal distance matrix: d(x,y) = min(||x-y||, ||x+y||)
    This respects projective identification x ~ -x.
    """
    print("Computing antipodal distance matrix...")
    D1 = pairwise_distances(Y, Y, metric='euclidean')
    D2 = pairwise_distances(Y, -Y, metric='euclidean')
    return np.minimum(D1, D2)


def compute_global_persistence(D, outdir):
    """Compute Vietoris-Rips persistence and plot diagrams."""
    if ripser is None:
        print("⚠ Skipping global persistence (ripser not installed)")
        return None
    
    print("\n" + "="*60)
    print("GLOBAL PERSISTENT HOMOLOGY (Vietoris-Rips)")
    print("="*60)
    
    print("Computing VR persistence up to H2...")
    r = ripser(D, distance_matrix=True, maxdim=2, n_perm=None)
    dgms = r['dgms']
    
    print(f"\nPersistence diagrams computed:")
    print(f"  H0: {len(dgms[0])} features")
    print(f"  H1: {len(dgms[1])} features")
    print(f"  H2: {len(dgms[2])} features")
    
    # Analyze persistence
    if len(dgms[0]) > 0:
        H0_pers = dgms[0][:, 1] - dgms[0][:, 0]
        H0_pers = H0_pers[~np.isinf(H0_pers)]
        print(f"\n  H0 interpretation: {len(H0_pers)+1} connected component(s)")
    
    if len(dgms[1]) > 0:
        H1_pers = dgms[1][:, 1] - dgms[1][:, 0]
        H1_robust = np.sum(H1_pers > np.percentile(H1_pers, 80))
        print(f"  H1 interpretation: {H1_robust} robust 1-cycles")
        print(f"      (Expected ~2 for torus-like real locus)")
    
    if len(dgms[2]) > 0:
        H2_pers = dgms[2][:, 1] - dgms[2][:, 0]
        H2_robust = np.sum(H2_pers > np.percentile(H2_pers, 80))
        print(f"  H2 interpretation: {H2_robust} robust 2-cycles")
        print(f"      (Expected ~1 for torus)")
    
    # Plot persistence diagrams
    print("\nGenerating persistence diagram...")
    plt.figure(figsize=(15, 4))
    plot_diagrams(dgms, show=False)
    plt.suptitle("Persistence Diagrams (Vietoris-Rips)", fontsize=14, y=1.02)
    outpath = os.path.join(outdir, "tda_persistence_diagrams.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {outpath}")
    plt.close()
    
    # Plot barcode
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    for dim in range(3):
        ax = axes[dim]
        dgm = dgms[dim]
        if len(dgm) == 0:
            continue
        # Sort by persistence
        pers = dgm[:, 1] - dgm[:, 0]
        order = np.argsort(pers)[::-1]
        dgm_sorted = dgm[order]
        
        for i, (birth, death) in enumerate(dgm_sorted[:50]):  # Top 50
            if np.isinf(death):
                death = birth + 2 * np.max(dgm_sorted[~np.isinf(dgm_sorted[:, 1]), 1])
            ax.plot([birth, death], [i, i], 'b-', linewidth=1.5)
        
        ax.set_xlabel("Filtration value", fontsize=10)
        ax.set_ylabel("Feature index", fontsize=10)
        ax.set_title(f"H{dim} Barcode", fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    outpath = os.path.join(outdir, "tda_persistence_barcodes.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {outpath}")
    plt.close()
    
    return dgms


def detect_singular_locus(X, D, k=50, outdir=None):
    """
    Detect singular locus via local homology and PCA.
    Returns: singular_flags (boolean array), local_scores, dim_ratio
    """
    if ripser is None:
        print("⚠ Skipping singular locus detection (ripser not installed)")
        return None, None, None
    
    print("\n" + "="*60)
    print("SINGULAR LOCUS DETECTION (Local Homology + PCA)")
    print("="*60)
    print(f"Analyzing neighborhoods with k={k} nearest neighbors...")
    
    # Find k-NN in antipodal metric
    nbrs = NearestNeighbors(n_neighbors=k, metric='precomputed').fit(D)
    dists, idxs = nbrs.kneighbors(D)
    
    n_points = X.shape[0]
    local_scores = np.zeros(n_points)  # Local H1 persistence
    dim_ratio = np.zeros(n_points)     # PCA dimension ratio
    
    print("Processing neighborhoods (this may take a while)...")
    for i in range(n_points):
        if i % 200 == 0:
            print(f"  Progress: {i}/{n_points} points...")
        
        neigh = X[idxs[i]]
        
        # Local VR persistence
        Dloc = pairwise_distances(neigh, neigh, metric='euclidean')
        try:
            rloc = ripser(Dloc, distance_matrix=True, maxdim=1)
            H0, H1 = rloc['dgms'][0], rloc['dgms'][1]
            # Total H1 persistence (should be ~1 strong class for regular points)
            pers_H1 = np.sum(H1[:, 1] - H1[:, 0]) if len(H1) > 0 else 0.0
            local_scores[i] = pers_H1
        except:
            local_scores[i] = 0.0
        
        # PCA dimension analysis
        pca = PCA(n_components=min(5, X.shape[1]))
        pca.fit(neigh)
        svals = pca.singular_values_
        # Ratio of 2nd to 1st singular value (small = 1D-like)
        dim_ratio[i] = (svals[1] / svals[0]) if len(svals) > 1 and svals[0] > 1e-12 else 0.0
    
    print(f"✓ Completed neighborhood analysis\n")
    
    # Identify singular candidates
    thr_pers = np.percentile(local_scores, 20)
    thr_dimr = np.percentile(dim_ratio, 20)
    singular_flags = (local_scores < thr_pers) | (dim_ratio < thr_dimr)
    
    print(f"Singular locus statistics:")
    print(f"  Local H1 persistence threshold: {thr_pers:.4f}")
    print(f"  PCA dimension ratio threshold: {thr_dimr:.4f}")
    print(f"  Singular-locus candidates: {np.sum(singular_flags)}/{n_points} ({100*np.mean(singular_flags):.1f}%)")
    print(f"  Expected: 1D curve (rational normal quartic ν₄(P¹))")
    
    if outdir:
        # Plot histograms
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].hist(local_scores, bins=50, alpha=0.7, edgecolor='black')
        axes[0].axvline(thr_pers, color='red', linestyle='--', linewidth=2, label=f'Threshold (20%ile)')
        axes[0].set_xlabel("Local H1 Persistence", fontsize=11)
        axes[0].set_ylabel("Frequency", fontsize=11)
        axes[0].set_title("Local Homology Score", fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(dim_ratio, bins=50, alpha=0.7, edgecolor='black', color='green')
        axes[1].axvline(thr_dimr, color='red', linestyle='--', linewidth=2, label=f'Threshold (20%ile)')
        axes[1].set_xlabel("PCA Dimension Ratio (σ₂/σ₁)", fontsize=11)
        axes[1].set_ylabel("Frequency", fontsize=11)
        axes[1].set_title("PCA Anisotropy", fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        outpath = os.path.join(outdir, "tda_singular_detection.png")
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {outpath}")
        plt.close()
    
    return singular_flags, local_scores, dim_ratio


def create_3d_embedding(X, singular_flags, outdir):
    """Create 3D UMAP embedding and visualizations."""
    if umap is None:
        print("⚠ Skipping 3D embedding (umap not installed)")
        return None
    
    print("\n" + "="*60)
    print("3D EMBEDDING & VISUALIZATION")
    print("="*60)
    print("Computing UMAP embedding to 3D...")
    
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=50,
        min_dist=0.05,
        metric='euclidean',
        random_state=42,
        verbose=False
    )
    emb = reducer.fit_transform(X)
    print(f"✓ Embedding computed: {emb.shape}")
    
    # Static matplotlib plot
    fig = plt.figure(figsize=(15, 5))
    
    # Regular points
    ax1 = fig.add_subplot(131, projection='3d')
    regular = ~singular_flags
    ax1.scatter(emb[regular, 0], emb[regular, 1], emb[regular, 2],
                c='steelblue', s=1, alpha=0.5, label='Regular')
    ax1.set_title("Regular Points", fontsize=11)
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    ax1.set_zlabel("UMAP 3")
    
    # Singular candidates
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(emb[singular_flags, 0], emb[singular_flags, 1], emb[singular_flags, 2],
                c='red', s=5, alpha=0.8, label='Singular')
    ax2.set_title("Singular Locus Candidates", fontsize=11)
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")
    ax2.set_zlabel("UMAP 3")
    
    # Combined
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(emb[regular, 0], emb[regular, 1], emb[regular, 2],
                c='steelblue', s=1, alpha=0.3, label='Regular')
    ax3.scatter(emb[singular_flags, 0], emb[singular_flags, 1], emb[singular_flags, 2],
                c='red', s=5, alpha=0.9, label='Singular')
    ax3.set_title("Combined View", fontsize=11)
    ax3.set_xlabel("UMAP 1")
    ax3.set_ylabel("UMAP 2")
    ax3.set_zlabel("UMAP 3")
    ax3.legend()
    
    plt.tight_layout()
    outpath = os.path.join(outdir, "tda_3d_embedding.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {outpath}")
    plt.close()
    
    # Interactive plotly visualization
    if px is not None:
        print("Creating interactive 3D visualization...")
        fig = go.Figure()
        
        # Regular points
        fig.add_trace(go.Scatter3d(
            x=emb[regular, 0],
            y=emb[regular, 1],
            z=emb[regular, 2],
            mode='markers',
            marker=dict(size=2, color='steelblue', opacity=0.5),
            name='Regular points'
        ))
        
        # Singular candidates
        fig.add_trace(go.Scatter3d(
            x=emb[singular_flags, 0],
            y=emb[singular_flags, 1],
            z=emb[singular_flags, 2],
            mode='markers',
            marker=dict(size=4, color='red', opacity=0.9),
            name='Singular locus candidates'
        ))
        
        fig.update_layout(
            title="3D UMAP Embedding of Triple-Root Surface",
            scene=dict(
                xaxis_title="UMAP 1",
                yaxis_title="UMAP 2",
                zaxis_title="UMAP 3"
            ),
            width=1000,
            height=800
        )
        
        outpath = os.path.join(outdir, "tda_embedding_interactive.html")
        fig.write_html(outpath)
        print(f"✓ Saved: {outpath}")
    
    return emb


def create_mesh(emb, singular_flags, outdir):
    """Create 3D mesh from embedding using Ball Pivoting and Alpha Complex."""
    print("\n" + "="*60)
    print("MESH GENERATION")
    print("="*60)
    
    # Method 1: Ball Pivoting with Open3D
    if o3d is not None:
        print("\n--- Method 1: Ball Pivoting (Open3D) ---")
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(emb.astype(np.float64))
        
        print("Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=30)
        
        # Ball pivoting algorithm
        print("Generating mesh with Ball Pivoting...")
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = [avg_dist * r for r in [1.5, 2.0, 3.0]]
        print(f"  Using radii: {[f'{r:.4f}' for r in radii]}")
        
        mesh_bpa = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
        
        print(f"✓ BPA Mesh: {len(mesh_bpa.vertices)} vertices, {len(mesh_bpa.triangles)} triangles")
        
        # Clean up mesh
        mesh_bpa.remove_duplicated_vertices()
        mesh_bpa.remove_duplicated_triangles()
        mesh_bpa.remove_degenerate_triangles()
        mesh_bpa.compute_vertex_normals()
        
        print(f"  After cleanup: {len(mesh_bpa.vertices)} vertices, {len(mesh_bpa.triangles)} triangles")
        
        # Save BPA mesh
        outpath = os.path.join(outdir, "tda_surface_mesh_bpa.ply")
        o3d.io.write_triangle_mesh(outpath, mesh_bpa)
        print(f"✓ Saved: {outpath}")
        
        # Also save as OBJ for easier viewing
        outpath_obj = os.path.join(outdir, "tda_surface_mesh_bpa.obj")
        o3d.io.write_triangle_mesh(outpath_obj, mesh_bpa)
        print(f"✓ Saved: {outpath_obj}")
        
        # Save singular locus candidates as separate point cloud
        if np.sum(singular_flags) > 0:
            singular_pcd = o3d.geometry.PointCloud()
            singular_pcd.points = o3d.utility.Vector3dVector(emb[singular_flags].astype(np.float64))
            # Color them red
            colors = np.tile([1.0, 0.0, 0.0], (np.sum(singular_flags), 1))
            singular_pcd.colors = o3d.utility.Vector3dVector(colors)
            
            outpath = os.path.join(outdir, "tda_singular_curve.ply")
            o3d.io.write_point_cloud(outpath, singular_pcd)
            print(f"✓ Saved: {outpath}")
    else:
        print("⚠ Open3D not available, skipping Ball Pivoting")
    
    # Method 2: Alpha Complex with GUDHI
    if gd is not None:
        print("\n--- Method 2: Alpha Complex (GUDHI) ---")
        print("Computing alpha complex...")
        
        # Create alpha complex
        alpha_complex = gd.AlphaComplex(points=emb.astype(np.float64))
        simplex_tree = alpha_complex.create_simplex_tree()
        
        print(f"✓ Alpha complex: {simplex_tree.num_vertices()} vertices, {simplex_tree.num_simplices()} simplices")
        
        # Compute persistence to find optimal filtration value
        simplex_tree.compute_persistence()
        
        # Extract triangles at a good filtration value
        # Use the birth time of the most persistent H2 feature (if exists)
        dgms = [simplex_tree.persistence_intervals_in_dimension(i) for i in range(3)]
        
        # Find a good alpha value: slightly after the birth of long-lived H1 features
        if len(dgms[1]) > 0:
            H1_births = dgms[1][:, 0]
            H1_deaths = dgms[1][:, 1]
            H1_pers = H1_deaths - H1_births
            # Use 80th percentile of persistent H1 births
            alpha_value = np.percentile(H1_births[H1_pers > np.percentile(H1_pers, 50)], 80)
            print(f"  Using alpha filtration value: {alpha_value:.4f}")
        else:
            # Fallback: use median of all 2-simplex filtration values
            filtrations = [simplex_tree.filtration(s) for s in simplex_tree.get_skeleton(2) if len(s[0]) == 3]
            alpha_value = np.median(filtrations) if filtrations else 0.1
            print(f"  Using median alpha value: {alpha_value:.4f}")
        
        # Extract 2-skeleton (triangles) below this filtration
        triangles = []
        vertices_set = set()
        for simplex, filt in simplex_tree.get_skeleton(2):
            if len(simplex) == 3 and filt <= alpha_value:
                triangles.append(simplex)
                vertices_set.update(simplex)
        
        print(f"✓ Extracted {len(triangles)} triangles with {len(vertices_set)} unique vertices")
        
        if len(triangles) > 0 and o3d is not None:
            # Create Open3D mesh from alpha complex triangles
            mesh_alpha = o3d.geometry.TriangleMesh()
            mesh_alpha.vertices = o3d.utility.Vector3dVector(emb.astype(np.float64))
            mesh_alpha.triangles = o3d.utility.Vector3iVector(np.array(triangles, dtype=np.int32))
            
            # Clean up
            mesh_alpha.remove_duplicated_vertices()
            mesh_alpha.remove_duplicated_triangles()
            mesh_alpha.remove_degenerate_triangles()
            mesh_alpha.remove_unreferenced_vertices()
            mesh_alpha.compute_vertex_normals()
            
            print(f"  After cleanup: {len(mesh_alpha.vertices)} vertices, {len(mesh_alpha.triangles)} triangles")
            
            # Save alpha mesh
            outpath = os.path.join(outdir, "tda_surface_mesh_alpha.ply")
            o3d.io.write_triangle_mesh(outpath, mesh_alpha)
            print(f"✓ Saved: {outpath}")
            
            outpath_obj = os.path.join(outdir, "tda_surface_mesh_alpha.obj")
            o3d.io.write_triangle_mesh(outpath_obj, mesh_alpha)
            print(f"✓ Saved: {outpath_obj}")
        elif len(triangles) > 0:
            print("⚠ Open3D not available, cannot save alpha mesh")
        else:
            print("⚠ No triangles found at chosen filtration value")
    else:
        print("⚠ GUDHI not available, skipping Alpha Complex")
    
    print(f"\n✓ Mesh generation complete")


def main():
    parser = argparse.ArgumentParser(
        description="TDA Analysis of Triple-Root Surface"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="../data/positive_samples.csv",
        help="Path to CSV file with positive samples"
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=50,
        help="Number of neighbors for local analysis (default: 50)"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="../plot",
        help="Output directory for plots (default: ../plot)"
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Subsample to N points (optional, for faster computation)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    print("="*60)
    print("TDA ANALYSIS: Triple-Root Surface in P⁴")
    print("="*60)
    print(f"CSV file: {args.csv}")
    print(f"Output directory: {args.outdir}")
    print(f"Neighbors: {args.neighbors}")
    print()
    
    # 1. Load and normalize data
    print("="*60)
    print("LOADING & PREPROCESSING DATA")
    print("="*60)
    
    Xraw = pd.read_csv(args.csv, header=None).values
    print(f"Loaded: {Xraw.shape[0]} points × {Xraw.shape[1]} dimensions")
    
    if Xraw.shape[1] == 10:
        print("Detected: Complex points in C⁵ (represented as R¹⁰)")
    elif Xraw.shape[1] == 5:
        print("Detected: Real points in R⁵")
    else:
        raise ValueError(f"Expected 5 or 10 columns, got {Xraw.shape[1]}")
    
    # Subsample if requested
    if args.subsample and args.subsample < Xraw.shape[0]:
        print(f"Subsampling to {args.subsample} points...")
        indices = np.random.choice(Xraw.shape[0], args.subsample, replace=False)
        Xraw = Xraw[indices]
    
    # Normalize to projective space
    print("Normalizing to projective space...")
    X = normalize_projective_rows(Xraw)
    print(f"✓ Normalized: {X.shape}")
    
    # Compute antipodal distance matrix
    D = antipodal_distances(X)
    print(f"✓ Antipodal distance matrix: {D.shape}")
    
    # 2. Global persistent homology
    dgms = compute_global_persistence(D, args.outdir)
    
    # 3. Detect singular locus
    singular_flags, local_scores, dim_ratio = detect_singular_locus(
        X, D, k=args.neighbors, outdir=args.outdir
    )
    
    if singular_flags is None:
        print("\n⚠ Singular locus detection skipped. Install ripser to enable.")
        singular_flags = np.zeros(X.shape[0], dtype=bool)
    
    # 4. Create 3D embedding
    emb = create_3d_embedding(X, singular_flags, args.outdir)
    
    # 5. Generate mesh
    if emb is not None:
        create_mesh(emb, singular_flags, args.outdir)
    
    print("\n" + "="*60)
    print("TDA ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nGenerated files in {args.outdir}:")
    print("  Persistence Analysis:")
    print("    - tda_persistence_diagrams.png")
    print("    - tda_persistence_barcodes.png")
    print("  Singular Locus Detection:")
    print("    - tda_singular_detection.png")
    print("  3D Visualizations:")
    print("    - tda_3d_embedding.png")
    print("    - tda_embedding_interactive.html")
    print("  Mesh Files:")
    print("    - tda_surface_mesh_bpa.ply/.obj (Ball Pivoting)")
    print("    - tda_surface_mesh_alpha.ply/.obj (Alpha Complex)")
    print("    - tda_singular_curve.ply (Singular locus candidates)")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
