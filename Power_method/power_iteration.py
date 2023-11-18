import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
    x = np.random.rand(data.shape[0])

    for _ in range(num_steps):
        # Multiply the matrix with the vector
        x = np.dot(data, x)
        # Normalize the vector
        x = x / np.linalg.norm(x)

    # Estimate the eigenvalue
    eigenvalue = np.dot(np.dot(x, data), x)

    return float(eigenvalue), x