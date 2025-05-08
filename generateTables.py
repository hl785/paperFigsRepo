"""Script for generating the LaTeX table with the parameters for each method."""
import numpy as np

def exchange_matrix(n):
    """Create an exchange matrix of size n x n
    
    :arg n: Size of the matrix
    """
    J = np.zeros((n, n))
    for i in range(n):
        J[i,n-i-1] = 1
    return J

def transform_params(gamma):
    """Transform the parameters gamma into the parameter lists alpha and beta.
    
    :arg gamma: List of untransformed parameters
    """
    K = len(gamma)+2
    if K % 2 == 0:
        K_alpha = (K-2)//2
        K_beta = (K-2)//2
    else:
        K_alpha = (K-1)//2
        K_beta = (K-3)//2

    gamma_alpha = np.array(gamma[:K_alpha])
    gamma_beta = np.array(gamma[K_alpha:])

    A = np.zeros((K, K_alpha))
    B = np.zeros((K, K_beta))
    C = np.zeros(K)
    D = np.zeros(K)

    if K % 2 == 0:
        A[:K_alpha,:] = np.eye(K_alpha)
        A[K_alpha:K_alpha+2,:] = -1
        A[K_alpha+2:,:] = exchange_matrix(K_alpha)
        B[:K_beta,:] = np.eye(K_beta)
        B[K_beta:K_beta+1,:] = -2
        B[K_beta+1:2*K_beta+1,:] = exchange_matrix(K_beta)
        C[K_alpha:K_alpha+2] = 1/2
        D[K_alpha] = 1
    else:
        A[:K_alpha,:] = np.eye(K_alpha)
        A[K_alpha:K_alpha+1,:] = -2
        A[K_alpha+1:,:] = exchange_matrix(K_alpha)
        B[:K_beta,:] = np.eye(K_beta)
        B[K_beta:K_beta+2,:] = -1
        B[K_beta+2:2*K_beta+2,:] = exchange_matrix(K_beta)
        C[K_alpha] = 1
        D[K_beta:K_beta+2] = 1/2
    
    alpha = A @ gamma_alpha + C
    beta = B @ gamma_beta + D    
    return alpha, beta

def check_symmetry(alpha,beta):
    """Check that the transformed parameters satisfy the symmetry condition.

    The parameter lists need to be palindromic.
    
    :arg alpha: potential flux parameters
    :arg beta: kinetic flux parameters
    """
    assert np.all(np.asarray(alpha) == np.asarray(alpha)[::-1])
    assert np.asarray(beta[-1]) == 0
    assert np.all(np.asarray(beta[:-1]) == np.asarray(beta[-2::-1]))

def check_consistency(alpha, beta):
    """Check that the transformed parameters satisfy the consistency condition.
    
    The entries in the parameter lists need to sum to 1.

    :arg alpha: potential flux parameters
    :arg beta: kinetic flux parameters
    """
    assert np.sum(np.asarray(alpha)) == 1
    assert np.sum(np.asarray(beta)) == 1

def format_list(lst):
    return "["+",".join(["0" if item==0 else f"{item:.3f}" for item in lst])+"]" 

def generate_latex_table(gamma, filename):
    """Generates a LaTeX table with the parameters for each method.

    :arg gamma: Dictionary with method names as keys and untransformed 
                parameter lists as values
    :arg filename: Name of the file to save the LaTeX table to
    """

    with open(filename,"w",encoding="utf-8") as f:
        print(r"\begin{tabular}{ |c|c|c| }",file=f) 
        print(r"\hline",file=f)
        print(r"Splitting & & coefficients \\ \hline \hline",file=f)
            
            
        
        for method, gamma in gamma.items():
            alpha, beta = transform_params(gamma)
            check_symmetry(alpha, beta)
            check_consistency(alpha, beta)
            print(r"\multirow{3}{16ex}{"+method+r"} ",file=f)
            print(r"& $\allParams$ & ",file=f)
            print(f"${format_list(gamma)}$"+r"\\", file=f)
            print(r"& $\potParams$ & ",file=f)
            print(f"${format_list(alpha)}$"+r"\\", file=f)
            print(r"& $\kinParams$ & ",file=f)
            print(f"${format_list(beta)}$"+r"\\", file=f)
            print(r"\hline",file=f)
        print(r"\end{tabular}",file=f)

###########################################################################
######### M A I N #########################################################
###########################################################################

if __name__ == "__main__":

    gamma = dict(Learn5A=[0.3627, -0.1003, -0.1353],
                 Learn5AProj=[0.346, -0.112, -0.132],
                 Learn8A=[0.2135, -0.0582, 0.4125, -0.1352, 0.4443, -0.0251],
                 Learn8B=[0.1178, 0.3876, 0.3660, 0.2922, 0.0564, -0.0212])
    generate_latex_table(gamma, "table_coefficients.tex")
