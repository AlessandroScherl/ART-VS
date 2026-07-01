import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import torch
from PIL import Image


def visualize_correspondences(image1, image2, points1, points2, save_path=None):
    """Visualize correspondences between two images."""
    if isinstance(image1, Image.Image):
        image1 = np.array(image1)
    if isinstance(image2, Image.Image):
        image2 = np.array(image2)

    if torch.is_tensor(points1):
        points1 = points1.cpu().detach().numpy()
    if torch.is_tensor(points2):
        points2 = points2.cpu().detach().numpy()

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow(image1)
    ax2.imshow(image2)

    ax1.axis('off')
    ax2.axis('off')

    # Handle empty points
    if len(points1) == 0:
        colors = []
    else:
        colors = plt.cm.plasma(np.linspace(0.05, 0.95, len(points1)))

    for i, ((y1, x1), (y2, x2), color) in enumerate(zip(points1, points2, colors)):
        ax1.plot(x1, y1, 'o', color=color, markersize=7, markeredgecolor='white', markeredgewidth=0.5)
        ax2.plot(x2, y2, 'o', color=color, markersize=7, markeredgecolor='white', markeredgewidth=0.5)

        con = ConnectionPatch(
            xyA=(x1, y1), xyB=(x2, y2),
            coordsA="data", coordsB="data",
            axesA=ax1, axesB=ax2, color=color, alpha=0.35
        )
        fig.add_artist(con)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def visualize_correspondences_ros(goal_image, current_image, points1, points2, matches,
                                bridge, correspondence_pub, goal_image_pub, current_image_pub,
                                use_tiling, tiling_config):
    """
    Create and publish visualization of correspondences between two images for ROS.

    Args:
        goal_image: PIL Image of the goal image
        current_image: PIL Image of the current image
        points1: numpy array or tensor of points in goal image
        points2: numpy array or tensor of points in current image
        matches: Optional list of match dictionaries with tile information
        bridge: CvBridge instance
        correspondence_pub: ROS publisher for correspondence visualization
        goal_image_pub: ROS publisher for goal image
        current_image_pub: ROS publisher for current image
        use_tiling: Whether tiling is enabled
        tiling_config: Tiling configuration object
    """
    import cv2

    # Fixed output dimensions for consistent video recording
    FIXED_WIDTH = 1200
    FIXED_HEIGHT = 600
    FIXED_DPI = 100

    # Create figure with fixed size and DPI
    fig = plt.figure(figsize=(FIXED_WIDTH / FIXED_DPI, FIXED_HEIGHT / FIXED_DPI), dpi=FIXED_DPI)

    # Use fixed subplot positions instead of tight_layout (prevents jiggle)
    ax1 = fig.add_axes([0.02, 0.02, 0.46, 0.96])  # [left, bottom, width, height]
    ax2 = fig.add_axes([0.52, 0.02, 0.46, 0.96])

    ax1.imshow(goal_image)
    ax2.imshow(current_image)

    # Convert points to numpy if they're tensors
    points1_np = points1.cpu().numpy() if torch.is_tensor(points1) else np.array(points1)
    points2_np = points2.cpu().numpy() if torch.is_tensor(points2) else np.array(points2)

    # Handle empty points case
    if len(points1_np) == 0 or len(points2_np) == 0:
        ax1.set_title("Goal Image", fontsize=10)
        ax2.set_title("Current Image (No matches)", fontsize=10)
    else:
        # Generate colors based on tiles if available
        if matches and use_tiling:
            colors = plt.cm.plasma(np.linspace(0.05, 0.95, tiling_config.slice_number))
            point_colors = [colors[m['tile_idx']] for m in matches]
        else:
            point_colors = plt.cm.plasma(np.linspace(0.05, 0.95, max(1, len(points1_np))))

        # Plot correspondences
        for i, ((y1, x1), (y2, x2), color) in enumerate(zip(points1_np, points2_np, point_colors)):
            ax1.plot(x1, y1, 'o', color=color, markersize=7, markeredgecolor='white', markeredgewidth=0.5)
            ax2.plot(x2, y2, 'o', color=color, markersize=7, markeredgecolor='white', markeredgewidth=0.5)

            # Draw correspondence lines
            con = ConnectionPatch(
                xyA=(x1, y1), xyB=(x2, y2),
                coordsA="data", coordsB="data",
                axesA=ax1, axesB=ax2, color=color, alpha=0.35
            )
            fig.add_artist(con)

        # Add grid lines if using tiling
        if use_tiling:
            for ax in [ax1, ax2]:
                for i in range(1, tiling_config.grid_size):
                    ax.axhline(y=i*tiling_config.slice_resolution, color='white', linewidth=1, linestyle='--', alpha=0.3)
                    ax.axvline(x=i*tiling_config.slice_resolution, color='white', linewidth=1, linestyle='--', alpha=0.3)

    # Set axis limits to match image dimensions
    ax1.set_xlim(0, goal_image.size[0])
    ax1.set_ylim(goal_image.size[1], 0)  # Invert y-axis for image coordinates
    ax2.set_xlim(0, current_image.size[0])
    ax2.set_ylim(current_image.size[1], 0)  # Invert y-axis for image coordinates

    ax1.axis('off')
    ax2.axis('off')

    # NO tight_layout() - we use fixed axes positions for consistent output size

    # Convert figure to image with fixed dimensions
    fig.canvas.draw()
    img_data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_data = img_data.reshape((FIXED_HEIGHT, FIXED_WIDTH, 4))[:, :, :3]  # Fixed dimensions, drop alpha

    # Convert RGB to BGR for ROS (cv_bridge expects BGR)
    img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
    ros_image = bridge.cv2_to_imgmsg(img_bgr, encoding="bgr8")

    # Publish to correspondence visualization topic
    correspondence_pub.publish(ros_image)

    plt.close(fig)


def visualize_correspondences_ros_cv2(goal_image, current_image, points1, points2, matches,
                                      bridge, correspondence_pub, goal_image_pub, current_image_pub,
                                      use_tiling, tiling_config):
    """Fast cv2 equivalent of visualize_correspondences_ros (no matplotlib figure render).

    Produces the same 1200x600 side-by-side overlay (goal | current) with plasma-coloured
    correspondence points + faint connecting lines, but draws with cv2 primitives instead of
    rendering a matplotlib canvas — typically 10-50x faster per frame. Selected when
    config.visualization_backend == 'cv2'. Signature matches visualize_correspondences_ros
    so the two are drop-in interchangeable.
    """
    import cv2

    PANEL_W, PANEL_H = 600, 600  # each panel; canvas = 1200x600 (matches matplotlib output)

    g = np.array(goal_image) if isinstance(goal_image, Image.Image) else np.asarray(goal_image)
    c = np.array(current_image) if isinstance(current_image, Image.Image) else np.asarray(current_image)
    gw, gh = (goal_image.size if isinstance(goal_image, Image.Image) else (g.shape[1], g.shape[0]))
    cw, ch = (current_image.size if isinstance(current_image, Image.Image) else (c.shape[1], c.shape[0]))

    # Panels (input is RGB -> convert to BGR for ROS bgr8)
    g_b = cv2.cvtColor(cv2.resize(g, (PANEL_W, PANEL_H)), cv2.COLOR_RGB2BGR)
    c_b = cv2.cvtColor(cv2.resize(c, (PANEL_W, PANEL_H)), cv2.COLOR_RGB2BGR)
    canvas = np.zeros((PANEL_H, PANEL_W * 2, 3), dtype=np.uint8)
    canvas[:, :PANEL_W] = g_b
    canvas[:, PANEL_W:] = c_b

    p1 = points1.cpu().numpy() if torch.is_tensor(points1) else np.array(points1)
    p2 = points2.cpu().numpy() if torch.is_tensor(points2) else np.array(points2)
    n = min(len(p1), len(p2))

    if n > 0:
        # plasma colours (matplotlib colormap LUT — cheap, no figure rendering)
        if matches and use_tiling:
            lut = (plt.cm.plasma(np.linspace(0.05, 0.95, tiling_config.slice_number))[:, :3] * 255).astype(int)
            pcolors = [lut[m['tile_idx']] for m in matches]
        else:
            pcolors = (plt.cm.plasma(np.linspace(0.05, 0.95, max(1, n)))[:, :3] * 255).astype(int)

        sx_g, sy_g = PANEL_W / float(gw), PANEL_H / float(gh)
        sx_c, sy_c = PANEL_W / float(cw), PANEL_H / float(ch)

        # Faint connection lines on an overlay, blended at alpha=0.4 (mimics matplotlib alpha)
        overlay = canvas.copy()
        pts = []
        for i in range(n):
            y1, x1 = p1[i]
            y2, x2 = p2[i]
            col = pcolors[i] if i < len(pcolors) else pcolors[-1]
            bgr = (int(col[2]), int(col[1]), int(col[0]))
            gx, gy = int(x1 * sx_g), int(y1 * sy_g)
            cx, cy = int(x2 * sx_c) + PANEL_W, int(y2 * sy_c)
            pts.append((gx, gy, cx, cy, bgr))
            cv2.line(overlay, (gx, gy), (cx, cy), bgr, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)

        # Solid points on top (coloured fill + white edge)
        for gx, gy, cx, cy, bgr in pts:
            for (px, py) in ((gx, gy), (cx, cy)):
                cv2.circle(canvas, (px, py), 4, bgr, -1, cv2.LINE_AA)
                cv2.circle(canvas, (px, py), 4, (255, 255, 255), 1, cv2.LINE_AA)

    correspondence_pub.publish(bridge.cv2_to_imgmsg(canvas, encoding="bgr8"))
