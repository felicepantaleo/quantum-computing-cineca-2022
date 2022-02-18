import numpy as np
import numpy.linalg as LA

def trueSVD(tensor, legs_left, legs_right, perm_left=None,
             perm_right=None, contract_singvals='N'):
        """Perform a Singular Value Decomposition by
        first reshaping the tensor into a legs_left x legs_right
        matrix, and permuting the legs of the ouput tensors if needed.

        Parameters
        ----------
        tensor : ndarray
            Tensor upon which apply the SVD
        legs_left : list of int
            Legs that will compose the rows of the matrix
        legs_right : list of int
            Legs that will compose the columns of the matrix
        perm_left : list of int, optional
            permutations of legs after the SVD on left tensor
        perm_right : list of int, optional
            permutation of legs after the SVD on right tensor
        contract_singvals: string, optional
            How to contract the singular values.
                'N' : no contraction
                'L' : to the left tensor
                'R' : to the right tensor

        Returns
        -------
        tens_left: ndarray
            left tensor after the SVD
        tens_right: ndarray
            right tensor after the SVD
        singvals: ndarray
            singular values kept after the SVD
        singvals_cutted: ndarray
            singular values cutted after the SVD, normalized with the biggest singval
        """

        # Reshaping
        matrix = np.transpose(tensor, legs_left+legs_right )
        shape_left = np.array(tensor.shape)[legs_left]
        shape_right = np.array(tensor.shape)[legs_right]
        matrix = matrix.reshape( np.prod(shape_left), np.prod(shape_right) )

        # SVD decomposition
        UU, singvals, Vh = LA.svd(matrix, full_matrices=False)

        # Contract singular values if requested
        if contract_singvals == 'L':
            UU = np.dot(UU, np.diag(singvals) )
        elif contract_singvals == 'R':
            Vh = np.dot(np.diag(singvals), Vh)
        elif contract_singvals != 'N':
            warn(f'contract_singvals option {contract_singvals} is not implemented')

        # Reshape back to tensors
        tens_left = UU.reshape( list(shape_left)+[len(singvals)] )
        if perm_left is not None:
            tens_left = np.transpose(tens_left, perm_left )

        tens_right = Vh.reshape( [len(singvals)]+list(shape_right) )
        if perm_right is not None:
            tens_right = np.transpose(tens_right, perm_right)

        return tens_left, tens_right

    

def tSVD(tensor, legs_left, legs_right, perm_left=None,
             perm_right=None, contract_singvals='N', max_bond_dim=10,
            cut_ratio=1e-9):
        """Perform a truncated Singular Value Decomposition by
        first reshaping the tensor into a legs_left x legs_right
        matrix, and permuting the legs of the ouput tensors if needed.

        Parameters
        ----------
        tensor : ndarray
            Tensor upon which apply the SVD
        legs_left : list of int
            Legs that will compose the rows of the matrix
        legs_right : list of int
            Legs that will compose the columns of the matrix
        perm_left : list of int, optional
            permutations of legs after the SVD on left tensor
        perm_right : list of int, optional
            permutation of legs after the SVD on right tensor
        contract_singvals: string, optional
            How to contract the singular values.
                'N' : no contraction
                'L' : to the left tensor
                'R' : to the right tensor
        max_bond_dim : int, optional
            Maximum bond dimension. Default to 10
        cut_ratio : float, optional
            Cut ratio. Default to 1e-9.

        Returns
        -------
        tens_left: ndarray
            left tensor after the SVD
        tens_right: ndarray
            right tensor after the SVD
        singvals: ndarray
            singular values kept after the SVD
        singvals_cutted: ndarray
            singular values cutted after the SVD, normalized with the biggest singval
        """

        # Reshaping
        matrix = np.transpose(tensor, legs_left+legs_right )
        shape_left = np.array(tensor.shape)[legs_left]
        shape_right = np.array(tensor.shape)[legs_right]
        matrix = matrix.reshape( np.prod(shape_left), np.prod(shape_right) )

        # SVD decomposition
        UU, singvals_tot, Vh = LA.svd(matrix, full_matrices=False)

        # Truncation
        lambda1 = singvals_tot[0]
        cut = np.nonzero(singvals_tot/lambda1 < cut_ratio)[0]
        if len(cut)>0:
            cut = min( max_bond_dim, cut[0] )
        else:
            cut = max_bond_dim
        cut = min(cut, len(singvals_tot))
        singvals = singvals_tot[:cut]
        UU = UU[:, :cut]
        Vh = Vh[:cut, :]
        singvals_cutted = singvals_tot[cut:]

        # Renormalizing the singular values vector to its norm
        # before the truncation
        norm_kept = np.sum(singvals**2)
        norm_trunc = np.sum(singvals_cutted**2)
        normalization_factor = np.sqrt(norm_kept)/np.sqrt(norm_kept + norm_trunc)
        singvals /= normalization_factor
        # Renormalize cut singular values to track the norm loss
        singvals_cutted /= np.sqrt( np.sum(singvals_tot**2) )

        # Contract singular values if requested
        if contract_singvals == 'L':
            UU = np.dot(UU, np.diag(singvals) )
        elif contract_singvals == 'R':
            Vh = np.dot(np.diag(singvals), Vh)
        elif contract_singvals != 'N':
            warn(f'contract_singvals option {contract_singvals} is not implemented')

        # Reshape back to tensors
        tens_left = UU.reshape( list(shape_left)+[cut] )
        if perm_left is not None:
            tens_left = np.transpose(tens_left, perm_left )

        tens_right = Vh.reshape( [cut]+list(shape_right) )
        if perm_right is not None:
            tens_right = np.transpose(tens_right, perm_right)

        return tens_left, tens_right, singvals, singvals_cutted