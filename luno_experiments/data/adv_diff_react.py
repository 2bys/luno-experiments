"""Generate various advection-diffusion-reaction trajectories for OOD experiments."""

import numpy as np
import scipy.signal
import scipy.integrate
from scipy.ndimage import gaussian_filter

from luno_experiments.enums import Data
from loguru import logger
import matplotlib.pyplot as plt

# -- Data Scenario Configs -----------------------------

AdvDiffReactScenarios = {
    Data.BASE_2,
    Data.FLIP_2,
    Data.POS_2,
    Data.POS_NEG_2,
    Data.POS_NEG_FLIP_2,
}

ScenarioConfig = {
    Data.BASE_2: {
        "n_shapes": "random",
        "side": "both",
        "heating_source": False,
        "cooling_term": False,
        "flip_velocity": False,
    },
    Data.FLIP_2: {
        "n_shapes": "random",
        "side": "both",
        "heating_source": False,
        "cooling_term": False,
        "flip_velocity": True,
    },
    Data.POS_2: {
        "n_shapes": "random",
        "side": "both",
        "heating_source": True,
        "cooling_term": False,
        "flip_velocity": False,
    },
    Data.POS_NEG_2: {
        "n_shapes": "random",
        "side": "both",
        "heating_source": True,
        "cooling_term": True,
        "flip_velocity": False,
    },
    Data.POS_NEG_FLIP_2: {
        "n_shapes": "random",
        "side": "both",
        "heating_source": True,
        "cooling_term": True,
        "flip_velocity": True,
    }
}

# ------------------------------------------------------
# -- Initial Condition Generators ----------------------
# ------------------------------------------------------

def generate_gaussian_blobs(X, Y, n_shapes="random", side='both', margin=0.1, sigma=2):
    """
    Generate initial conditions using random Gaussian blobs.

    Parameters
    ----------
    X, Y : ndarray
        2D meshgrid coordinates in [0,1]x[0,1]
    n_shapes : int or str, optional
        Number of Gaussian blobs to generate. If "random", generates between 1-10 blobs.
    side : str, optional
        Side of the domain to place blobs on. Options: 'left', 'right', or 'both'.
    margin : float, optional
        Minimum distance from domain boundaries for blob centers.
    sigma : float, optional
        Standard deviation for Gaussian smoothing of the final result.

    Returns
    -------
    ndarray
        2D array of shape (nx, ny) containing the initial conditions.
    """
    nx, ny = X.shape
    init = np.zeros((nx, ny))
    sides = ['left', 'right'] if side == 'both' else [side]
    if n_shapes == "random":
        n_shapes = np.random.randint(1, 10)
    for _ in range(n_shapes):
        s = np.random.choice(sides)
        cx = np.random.uniform(margin, 0.5-margin) \
            if s=='left' else np.random.uniform(0.5+margin, 1-margin)
        cy = np.random.uniform(margin, 1-margin)
        sig = np.random.uniform(0.02, 0.1)
        blob = np.exp(-((X-cx)**2 + (Y-cy)**2)/(2*sig**2))
        init += blob
    init = gaussian_filter(init, sigma=sigma, mode='constant', cval=0)
    init[0,:]=init[-1,:]=init[:,0]=init[:,-1]=0
    return init

def generate_triangle(X, Y, side="left", sigma=0.0):
    """
    Generate a single triangular shape as initial condition.

    Parameters
    ----------
    X, Y : ndarray
        2D meshgrid coordinates in [0,1]x[0,1]
    side : str, optional
        Side of the domain to place the triangle. Options: 'left', 'right', or 'both'.
        If 'both', randomly chooses between 'left' and 'right'.
    sigma : float, optional
        Standard deviation for Gaussian smoothing of the triangle edges.

    Returns
    -------
    ndarray
        2D array of shape (nx, ny) containing the triangular initial condition.
    """
    nx, ny = X.shape
    initial = np.zeros((nx, ny))

    if side == "both":
        side = np.random.choice(["left", "right"])

    if side == "left":
        # Triangle in lower-left
        x_corner = 0.0
    else:
        # Triangle in lower-right
        x_corner = 0.5

    # Set the bounds of the triangle
    x_min = x_corner
    x_max = x_corner + 0.4
    y_min = 0.2
    y_max = 0.6

    # For each point, check if it's under the linear boundary of the triangle
    # The line is: y - y_min = slope * (x - x_min)
    slope = (y_max - y_min) / (x_max - x_min)

    for i in range(nx):
        for j in range(ny):
            if X[i, j] >= x_min and X[i, j] <= x_max:
                # y_th is the linear interpolation from x_min to x
                y_th = y_min + slope * (X[i, j] - x_min)
                if (Y[i, j] >= y_min) and (Y[i, j] <= y_th):
                    initial[i, j] = 1.0

    # Apply Gaussian smoothing to the sharp edges
    smoothed = gaussian_filter(initial, sigma=sigma)

    return smoothed


def generate_smiley(X, Y, face_sigma=0.1, feature_sigma=0.018, **kwargs):
    """
    Generate a happy smiley face with random placement using Gaussian blobs.

    Parameters
    ----------
    X, Y : ndarray
        2D meshgrid coordinates in [0,1]x[0,1]
    face_sigma : float, optional
        Size of the main face circle
    feature_sigma : float, optional
        Size of the features (eyes and smile)
    **kwargs : dict
        Additional keyword arguments (unused)

    Returns
    -------
    ndarray
        2D array of shape (nx, ny) containing the smiley face initial condition.
    """
    del kwargs
    nx, ny = X.shape
    initial = np.zeros((nx, ny))

    # Randomize the center of the face
    cx = np.random.uniform(0.2, 0.8)  # Keep some margin from edges
    cy = np.random.uniform(0.2, 0.8)

    # Create rotation matrix for random rotation
    angle = np.random.uniform(0, np.pi)  # Random angle; only one direction.
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    def rotate(x, y):
        """Apply rotation transformation."""
        x_rot = cos_angle * (x - cx) - sin_angle * (y - cy) + cx
        y_rot = sin_angle * (x - cx) + cos_angle * (y - cy) + cy
        return x_rot, y_rot

    # Main face (larger Gaussian blob)
    X_rot, Y_rot = rotate(X, Y)
    face = np.exp(-((X_rot - cx) ** 2 + (Y_rot - cy) ** 2) / (2 * face_sigma**2))

    # Eyes (happy curved eyes - series of small Gaussian blobs)
    eye_offset_x = 0.08
    eye_offset_y = 0.1
    eye_points = 3
    eyes = np.zeros_like(initial)

    for i in range(eye_points):
        t = i / (eye_points - 1)
        # Curved eyes
        eye_y = cy + eye_offset_y - 0.01 * (2 * t - 1) ** 2  # upward parabola

        # Left eye
        eye_x = cx - eye_offset_x + 0.01 * (2 * t - 1)
        eye_x_rot, eye_y_rot = rotate(X, Y)
        eyes += np.exp(
            -((eye_x_rot - eye_x) ** 2 + (eye_y_rot - eye_y) ** 2)
            / (2 * feature_sigma**2)
        )

        # Right eye
        eye_x = cx + eye_offset_x + 0.02 * (2 * t - 1)
        eye_x_rot, eye_y_rot = rotate(X, Y)
        eyes += np.exp(
            -((eye_x_rot - eye_x) ** 2 + (eye_y_rot - eye_y) ** 2)
            / (2 * feature_sigma**2)
        )

    # Happy mouth (upward elliptical shape using Gaussian blobs)
    smile_points = 7
    smile_y_offset = -0.08
    smile_width = 0.08
    smile = np.zeros_like(initial)

    for i in range(smile_points):
        t = i / (smile_points - 1)
        smile_x = cx + smile_width * (2 * t - 1)
        # Upward curve for happy mouth
        smile_y = cy + smile_y_offset + 0.06 * (2 * t - 1) ** 2
        smile_x_rot, smile_y_rot = rotate(X, Y)
        smile += np.exp(
            -((smile_x_rot - smile_x) ** 2 + (smile_y_rot - smile_y) ** 2)
            / (2 * feature_sigma**2)
        )

    # Combine all features
    initial = 0.2 * face - 0.6 * (eyes + 0.3 * smile)

    # Clip negative values
    initial = np.maximum(initial, 0)

    # Normalize to [0,1] range
    initial = initial / initial.max()

    return initial


def sample_velocity_components(
    angle: np.ndarray | float,
    v_magnitude: np.ndarray | float,
    nx: int,
    ny: int,
    flip_half: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create velocity field components for given angles and magnitudes.

    Parameters
    ----------
    angle : ndarray or float
        Angle(s) in radians for the velocity direction(s)
    v_magnitude : ndarray or float
        Magnitude(s) of the velocity
    nx, ny : int
        Grid dimensions
    flip_half : bool, optional
        If True, flips the velocity direction for half of the domain (x > 0.5)

    Returns
    -------
    tuple[ndarray, ndarray]
        x and y components of velocity field, shape (n_samples, nx, ny) if batched
        or (nx, ny) if single sample
    """
    # Handle both single values and arrays
    if isinstance(angle, (int, float)):
        angle = np.array([angle])
        v_magnitude = np.array([v_magnitude])
        single_sample = True
    else:
        single_sample = False

    n_samples = len(angle)
    
    # Create constant velocity fields with given magnitudes and directions
    vx = np.zeros((n_samples, nx, ny))
    vy = np.zeros((n_samples, nx, ny))
    
    for i in range(n_samples):
        vx[i] = v_magnitude[i] * np.cos(angle[i]) * np.ones((nx, ny))
        vy[i] = v_magnitude[i] * np.sin(angle[i]) * np.ones((nx, ny))

    if flip_half:
        # Create a mask for the right half of the domain
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)
        mask = X > 0.5
        
        # Flip velocity components in the right half for all samples
        vx[:, mask] *= -1
        vy[:, mask] *= -1

    # Return single sample without batch dimension if input was scalar
    if single_sample:
        return vx[0], vy[0]
    return vx, vy


# -------------------------------------------------------
# -- Advection-Diffusion-Reaction Solver ----------------
# -------------------------------------------------------

def solve_advection_diffusion_reaction(
        ICs, 
        velocities,
        alpha=0.026,
        reaction_terms=None,
        dt=1e-3,
        nt=100,
        dh=1.0/100,
        method='RK45',
    ):
    """
    Solve the advection-diffusion-reaction equation using a finite difference method.

    Parameters
    ----------
    ICs : ndarray
        Initial conditions for each sample
    velocities : ndarray
        Velocity fields for each sample
    alpha : float, optional
        Diffusion coefficient
    reaction_terms : ndarray, optional
        Reaction terms for each sample
    dt : float, optional
        Time step size
    nt : int, optional
        Number of time steps
    dh : float, optional
        Grid spacing
    method : str, optional
        Integration method for solve_ivp

    Returns
    -------
    ndarray
        Array of solutions for each sample, shape (n_samples, nt, nx, ny)
    """
    n = len(ICs)
    if reaction_terms is None:
        reaction_terms = [np.zeros_like(ICs[0])]*n

    # Precompute stencils
    lap0 = np.array([[0,1,0],[1,-4,1],[0,1,0]],float)
    lap1 = np.array([[.5,0,.5],[0,-2,0],[.5,0,.5]],float)
    gamma = 0.5
    stencil_diff = ((1-gamma)*lap0 + gamma*lap1)/(dh**2)
    adv_x = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])/(6*dh)
    adv_y = adv_x.T

    def vector_field(t, u, vfield, react):
        u2 = u.reshape(ICs[0].shape)
        up = np.pad(u2,1)
        diff = scipy.signal.convolve(up, stencil_diff, mode='valid')
        ax = scipy.signal.convolve(up, adv_x, mode='valid')
        ay = scipy.signal.convolve(up, adv_y, mode='valid')
        vx, vy = vfield
        dudt = alpha*diff + vx*ax - vy*ay + react
        return dudt.ravel()

    results = []
    for ic, vf, react in zip(ICs, velocities, reaction_terms):
        sol = scipy.integrate.solve_ivp(
            vector_field, (0, dt*nt), ic.ravel(), method=method,
            t_eval=np.linspace(0,dt*nt,nt), args=(vf, react)
        )
        traj = sol.y.T.reshape(nt, *ic.shape)
        results.append(traj)
    return np.array(results)


# -------------------------------------------------------
# -- Data Generation -----------------------------------
# -------------------------------------------------------

def build_split(data_name, data_root_dir, mode, n_samples):
    """
    Build a single split (train/valid/test) of the advection-diffusion-reaction dataset.

    Parameters
    ----------
    data_name : Data
        Name of the dataset scenario
    data_root_dir : Path
        Root directory for saving the data
    mode : str
        Split mode ('train', 'valid', or 'test')
    n_samples : int
        Number of samples to generate

    Returns
    -------
    None
        Saves the generated data to disk and creates visualization
    """
    logger.info(f"Building {mode} split for {data_name} with {n_samples} samples...")

    # Build grid
    nx, ny = 100, 100
    logger.debug(f"Creating {nx}x{ny} grid")
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    # Get configuration for the current mode
    cfg = ScenarioConfig[data_name]
    logger.debug(f"Using configuration: {cfg}")

    # Sample initial conditions
    logger.info("Generating initial conditions...")
    ics = np.stack(
        [
            generate_gaussian_blobs(
                X, Y, n_shapes='random', side='both', margin=0.1, sigma=2
            )
            for _ in range(n_samples)
        ],
        axis=0
    )
    logger.debug(f"Generated initial conditions with shape {ics.shape}")

    # Sample velocity fields
    logger.info("Generating velocity fields...")
    angles = np.random.uniform(0, 2 * np.pi, size=n_samples)
    vel_magnitudes = np.random.uniform(1.0, 4.0, size=n_samples)
    logger.debug(f"Velocity angles range: [{angles.min():.2f}, {angles.max():.2f}]")
    logger.debug(f"Velocity magnitudes range: [{vel_magnitudes.min():.2f}, {vel_magnitudes.max():.2f}]")
    
    vx, vy = sample_velocity_components(
        angle=angles,
        v_magnitude=vel_magnitudes,
        nx=nx,
        ny=ny,
        flip_half=cfg["flip_velocity"],
    )
    velocities = np.stack([np.stack([vx[i], vy[i]]) for i in range(n_samples)])
    logger.debug(f"Generated velocity fields with shape {velocities.shape}")

    # Sample reaction terms
    logger.info("Generating reaction terms...")
    reaction_terms = np.zeros_like(ics)
    if cfg["heating_source"]:
        logger.debug("Adding heating source")
        reaction_terms += 10. * generate_triangle(X, Y, side='random', sigma=0.0)
    if cfg["cooling_term"]:
        logger.debug("Adding cooling term")
        reaction_terms -= 20. * generate_triangle(X, Y, side='random', sigma=0.0)
    logger.debug(f"Generated reaction terms with shape {reaction_terms.shape}")

    # Solve the advection-diffusion-reaction equation
    logger.info("Solving advection-diffusion-reaction equation...")
    solutions = solve_advection_diffusion_reaction(
        ICs=ics,
        velocities=velocities,
        reaction_terms=reaction_terms,
        dt=1e-3,
        nt=200,
    )
    trajs = np.stack(solutions[:, ::2])[:, :60] # sub-sample grid
    logger.debug(f"Generated trajectories with shape {trajs.shape}")
    
    # Create output directory if it doesn't exist
    out_dir = (data_root_dir / f"{data_name}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Saving data to {out_dir}")

    np.savez(
        out_dir / f"{mode}_solutions.npz",
        ICs=ics,
        velocities=velocities,
        reactions_terms=reaction_terms,
        trajectories=trajs,
    )
    logger.info(f"Saved solutions to {out_dir}/{mode}_solutions.npz")

    # Create visualization
    logger.info("Generating example visualization...")
    n_cols = min(n_samples, 4)
    fig, axes = plt.subplots(3, n_cols, figsize=(16, 12))
    for i in range(n_cols):
        # First row: initial conditions
        axes[0, i].imshow(ics[i])
        axes[0, i].set_title("Initial Condition")
        axes[0, i].axis("off")

        # Second row: first time step
        axes[1, i].imshow(trajs[i, 0])
        axes[1, i].set_title("First Time Step")
        axes[1, i].axis("off")

        # Third row: last time step  
        axes[2, i].imshow(trajs[i, -1])
        axes[2, i].set_title("Last Time Step")
        axes[2, i].axis("off")

    plt.savefig(out_dir / f"{mode}_examples.png")
    plt.close()
    logger.info(f"Saved visualization to {out_dir}/{mode}_examples.png")


def generate_adv_diff_react_data(args):
    """
    Generate the complete advection-diffusion-reaction dataset.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing:
        - data_name: Name of the dataset scenario
        - data_root_dir: Root directory for saving the data
        - num_train_samples: Number of training samples
        - num_valid_samples: Number of validation samples
        - num_test_samples: Number of test samples
        - train_seed: Random seed for training set
        - valid_seed: Random seed for validation set
        - test_seed: Random seed for test set

    Returns
    -------
    None
        Generates and saves the complete dataset
    """
    logger.info(f"Starting data generation for {args.data_name}")
    logger.debug(f"Arguments: {args}")
    
    # create folder path
    out_dir = (args.data_root_dir / f"{args.data_name}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Output directory: {out_dir}")
    
    # map splits to total counts and seeds
    splits = {
        "train": (args.num_train_samples, args.train_seed),
        "valid": (args.num_valid_samples, args.valid_seed),
        "test": (args.num_test_samples, args.test_seed),
    }
    logger.debug(f"Split configuration: {splits}")

    for split_name, (total_samples, seed) in splits.items():
        if total_samples == 0:
            logger.warning(f"Skipping {split_name} split as total samples is 0")
            continue
        logger.info(f"Setting random seed {seed} for {split_name} split")
        np.random.seed(seed)
        build_split(args.data_name, args.data_root_dir, split_name, total_samples)

    logger.info("Data generation complete.")
