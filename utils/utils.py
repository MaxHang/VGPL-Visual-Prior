import numpy as np

from mpl_toolkits.mplot3d import Axes3D  # Register 3d projection
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


DEFAULT_DPI = 100.0
BASIC_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
HIGHTLIGHT_COLORS = ['red', 'blue', 'yellow', 'black', 'green']

# Blender to Matplotlib conversion
blender_elev = 70  # Rotation X in Blender
blender_azim = 307  # Rotation Z in Blender

# Convert Blender rotation to matplotlib angles
matplotlib_elev = 90 - blender_elev  # Elev is the angle from the xy-plane upwards
matplotlib_azim = blender_azim - 90  # Azim is the angle around the z-axis from the positive y-axis

# def getColor(groups, use_max=True):
#     """
#     Inputs:
#         groups: (N, G)
#     Returns:
#         a list of colors
#     """
#     if use_max:
#         groups = np.argmax(groups, -1)
#         return [BASIC_COLORS[int(groups[i])] for i in range(groups.shape[0])]
#     else:
#         g_min = np.amin(groups)
#         groups -= g_min
#         raise NotImplementedError

def getColor(groups, use_max=True):
    """
    Inputs:
        groups: (N, G)
    Returns:
        a list of colors
    """
    if use_max:
        # print("11111111111111111111111111")
        # print(groups)
        # print(groups.shape)
        groups = np.zeros((6000,))

        # print(groups)

        return [BASIC_COLORS[int(groups[i])] for i in range(groups.shape[0])]

        # return ['tab:blue']
    else:
        g_min = np.amin(groups)
        groups -= g_min
        raise NotImplementedError


blender_camera = {
    'location': {'x': -3.7965, 'y': -2.8294, 'z': 2.7015},
    'rotation': {'x': 70, 'y': 0, 'z': 307},
    'scale': {'x': 0.5, 'y': 0.5, 'z': 0.5}
}

def match_blender_view(ax, blender_camera):
    # Extract Blender camera parameters
    location = blender_camera['location']
    rotation = blender_camera['rotation']
    scale = blender_camera['scale']

    # Convert Blender rotation to matplotlib angles
    # The conversion below is just a starting point, based on the provided images
    matplotlib_elev = 90 - rotation['x']  # Assuming Blender's rotation.x is equivalent to elev
    matplotlib_azim = rotation['z'] - 90  # Assuming Blender's rotation.z is equivalent to azim
    
    # Adjust the view
    ax.view_init(elev=matplotlib_elev, azim=matplotlib_azim)
    
    # Set axes limits proportionally to Blender's location and scale
    # This is a rough approximation, as the scale in Blender may not translate directly
    ax.set_xlim(location['x'] * scale['x'], location['x'] + scale['x'])
    ax.set_ylim(location['y'] * scale['y'], location['y'] + scale['y'])
    ax.set_zlim(location['z'] * scale['z'], location['z'] + scale['z'])


def toy_render(point_cloud,
               shape=None,
               title=None,
               highlight_idxs=[],
               groups=None,
               gt_groups=None,
               # xlim=(-0.7, 0.7),
               # ylim=(-0.7, 0.7),
               # zlim=(0, 0.7)):
               # 这是自己改的
            #    xlim=(-1, 1),
            #    ylim=(-1, 1),
            #    zlim=( 0, 4)):
               xlim=(-1, 1),
               ylim=( 0, 4),
               zlim=(-1, 1)):
    """
    Inputs:
        point_cloud: (T, N, 3)
        groups: (T, N, G)
        gt_groups: (N,)
    Returns:
        a list of frames [img1, img2, ...]
    """
    if shape is not None:
        figsize = (shape[0] / DEFAULT_DPI,
                   shape[1] / DEFAULT_DPI)
    else:
        figsize = None

    frames = []
    for i_f in range(point_cloud.shape[0]):
        xs = point_cloud[i_f, :, 0]
        zs = point_cloud[i_f, :, 1]
        ys = point_cloud[i_f, :, 2]

        if gt_groups is not None:
            # gt_groups (N,)
            color = [BASIC_COLORS[gt_groups[i]]
                     for i in range(gt_groups.shape[0])]
        elif groups is not None:
            # groups (T, N, G)
            color = getColor(groups[i_f].squeeze())
        else:
            # groups is None and gt_groups is None:
            color = ['gray' for _ in range(point_cloud.shape[1])]

        for j, hl_id in enumerate(highlight_idxs):
            color[hl_id] = HIGHTLIGHT_COLORS[j % len(HIGHTLIGHT_COLORS)]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=90, azim=-90)
        # ax.view_init(elev=30, azim=-155)
        ax.scatter(xs, ys, zs, color=color)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        if title is not None:
            ax.set_title(title)
        
        # 绘制x轴
        ax.plot([xlim[0], xlim[1]], [0, 0], [0, 0], color='red', linewidth=2)
        ax.text(xlim[1], 0, 0, 'X', color='red')

        # 绘制y轴
        ax.plot([0, 0], [ylim[0], ylim[1]], [0, 0], color='green', linewidth=2)
        ax.text(0, ylim[1], 0, 'Y', color='green')

        # 绘制z轴
        ax.plot([0, 0], [0, 0], [zlim[0], zlim[1]], color='blue', linewidth=2)
        ax.text(0, 0, zlim[1], 'Z', color='blue')

        fig.canvas.draw()

        frame = np.fromstring(
            fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)

        plt.close(fig)

    return np.stack(frames)


def plot_curves(x, ys,
                save_path=None,
                curve_labels=[],
                x_label='',
                y_label='',
                title=''):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, y in enumerate(ys):
        if i < len(curve_labels):
            ax.plot(x, y, label=curve_labels[i])
        else:
            ax.plot(x, y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path)
