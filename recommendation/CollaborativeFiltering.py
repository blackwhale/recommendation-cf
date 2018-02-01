import logging
import numpy as np

from progress.bar import Bar


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class CollaborativeFiltering:
    """This module is to compute Collaborative Filtering of a matrix
    supporting row and column (user and item) based method
    """
    BAR = None

    def __init__(self, input_mat, similarity='cosine'):
        """
        Args:
            input_mat (numpy.ndarray)
        Kwargs:
            similarity (string): only support 'cosine' and 'pearson'
                                 default value is 'cosine'
        """
        self.input_mat = input_mat
        self.input_mat_T = input_mat.T
        self.sim_func = self._set_similarity(similarity)
        self.shape = input_mat.shape
        self.column_sim = np.zeros((self.shape[1], self.shape[1]))
        self.row_sim = np.zeros((self.shape[0], self.shape[0]))

    def _gen_sim_mat(self, shape, top_n, transpose=False):
        """To generate similarity matrix
        Args:
            shape (tuple of two int): the shape of the similarity matrix
            top_n (int): how many top similarities should be considered
        Kwargs:
            transpose (bool): whether to transpose the output matrix or not
        Return:
            A similarity matrix with shape of input
        """
        sim_mat = np.zeros(shape)
        length = sim_mat.shape[0]

        # compute all similarities of row i and j
        logger.info('Start to compute similarity matrix')
        sim_mat = self.sim_func(transpose=transpose)

        # only keep top N elements in a row
        logger.info('Keep top {0} elements only'.format(top_n))
        if top_n:
            for i in xrange(length):
                indices = np.argsort(sim_mat[i, :])[:length-top_n]
                for j in indices:
                    sim_mat[i][j] = 0.0

        logger.info('Finish computing similarity matrix')
        return sim_mat if not transpose else sim_mat.T

    def row_based(self, top_n=0):
        """Do row based (user based) collaborative filtering
        Kwargs:
            top_n (int): how many similarities should be considered
                         to interpolate the rank(i, j)
                         0 means all elements are considered
        Return:
            A matrix of rankings
        """
        logger.info('Start to compute row based collaborative filtering')
        processing_steps = self.shape[0]*self.shape[1]

        self.BAR = Bar('Processing',
                       max=processing_steps,
                       suffix='%(percent).2f%% - %(eta_td)s remaining')

        # compute row-based similarity matrix
        self.row_sim = self._gen_sim_mat(self.row_sim.shape,
                                         top_n,
                                         transpose=False)

        # Do matrix multiplication to generate rankings
        input_mask = (self.input_mat == 0)
        computed_mat = np.dot(self.row_sim, self.input_mat)

        # Only non-ranked element(i, j) will be replaced
        logger.info('Start to compute adaptive weighting')
        output_matrix = np.zeros(self.shape)
        for i in xrange(self.shape[0]):
            for j in xrange(self.shape[1]):
                if input_mask[i, j]:
                    # adaptive weight for a row
                    weight = np.sum(
                        np.ma.array(
                            self.row_sim[i, :],
                            mask=input_mask[:, j]
                        )
                    )
                    value = (np.float32(0.0)
                             if np.isclose(weight, np.float32(0.0))
                             else computed_mat[i, j]/weight)
                    output_matrix[i, j] = value
                else:
                    output_matrix[i, j] = self.input_mat[i, j]
                self.BAR.next()

        self.BAR.finish()
        logger.info('Finish computing adaptive weighting')

        return np.around(output_matrix, decimals=3)

    def column_based(self, top_n=0):
        """Do column based (item based) collaborative filtering
        Kwargs:
            top_n (int): how many similarities should be considered
                         to interpolate the rank(i, j)
        Return:
            A matrix of rankings
        """
        logger.info('Start to compute column based collaborative filtering')
        processing_steps = self.shape[0]*self.shape[1]
        self.BAR = Bar('Processing',
                       max=processing_steps,
                       suffix='%(percent).2f%% - %(eta_td)s remaining')

        # compute column-based similarity matrix
        self.column_sim = self._gen_sim_mat(self.column_sim.shape,
                                            top_n,
                                            transpose=True)
        # Do matrix multiplication to generate rankings
        input_mask = (self.input_mat == 0)
        computed_mat = np.dot(self.input_mat, self.column_sim)

        # Only non-ranked element(i, j) will be replaced
        logger.info('Start to compute adaptive weighting')
        output_matrix = np.zeros(self.shape)
        for i in xrange(self.shape[0]):
            for j in xrange(self.shape[1]):
                if input_mask[i, j]:
                    # adaptive weight for a column
                    weight = np.sum(
                        np.ma.array(
                            self.column_sim[:, j],
                            mask=input_mask[i, :]
                        )
                    )
                    value = (np.float32(0.0)
                             if weight == np.float32(0.0)
                             else computed_mat[i, j]/weight)
                    output_matrix[i, j] = value.round(3)
                else:
                    output_matrix[i, j] = self.input_mat[i, j]
                self.BAR.next()

        self.BAR.finish()
        logger.info('Finish computing adaptive weighting')

        return output_matrix

    def _set_similarity(self, similarity):
        """Set similarity method
        Args:
            similarity (string): 'cosine' or 'pearson'
        Return:
            A similarity function
        """
        if similarity == 'cosine':
            logger.info('cosine similarity is chosen')
            return self._cosine_similarity
        if similarity == 'pearson':
            logger.info('pearson similarity is chosen')
            return self._pearson_similarity

        logger.error('Only support cosine and pearson similarities')
        raise ValueError('Only support cosine and pearson similarities')

    def _cosine_similarity(self, transpose=False):
        if transpose:
            similarity = np.dot(self.input_mat_T, self.input_mat)
        else:
            similarity = np.dot(self.input_mat, self.input_mat_T)

        square_mag = np.diag(similarity)
        inv_square_mag = 1 / square_mag
        inv_square_mag[np.isinf(inv_square_mag)] = 0
        inv_mag = np.sqrt(inv_square_mag)
        cosine = similarity * inv_mag
        cosine = cosine.T * inv_mag
        return cosine

    def _pearson_similarity(self, transpose=False):
        if transpose:
            covariance = np.cov(self.input_mat_T)
        else:
            covariance = np.cov(self.input_mat)
        diag = np.diag(covariance)
        sqrt = np.sqrt(diag)
        square = np.outer(sqrt, sqrt)
        pearson = covariance / square
        return pearson
