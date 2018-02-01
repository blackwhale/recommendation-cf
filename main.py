from datetime import timedelta
import logging
import time

import log_config

from utils import load_data, save_data
from recommendation import CollaborativeFiltering


logger = logging.getLogger(__name__)
logging.config.dictConfig(log_config.LOG_CONFIG)


def main():
    lr = 1
    lc = 2
    topn = 5
    #input_file = './data/test.csv'
    #input_file = './data/test/SimuData-item-300x20x.5-20180129-112918.csv'
    input_file = './data/test/SimuData-item-3000x200x.6-20180129-112920.csv'
    #input_file = './data/test/SimuData-item-10000x1500x.7-20180129-113005.csv'

    start = time.time()
    logger.info('Start to load data from {0}'.format(input_file))
    input_mat = load_data(input_file,
                          leading_row=lr, leading_column=lc)
    end = time.time()
    sec = int(end-start)
    logger.info('Finish loading data in {0} s'.format(timedelta(seconds=sec)))

    cf = CollaborativeFiltering(input_mat, similarity='cosine')
    print 'input matrix:\n', input_mat

    start = time.time()
    rb = cf.row_based(top_n=topn)
    end = time.time()
    sec = int(end-start)
    logger.info('User-based elapsed {0}'.format(timedelta(seconds=sec)))

    print 'row based:\n', rb
    save_data(input_file, './rb.csv', rb, leading_row=lr, leading_column=lc)

    start = time.time()
    cb = cf.column_based(top_n=topn)
    end = time.time()
    sec = int(end-start)
    logger.info('Item-based elapsed {0}'.format(timedelta(seconds=sec)))

    print 'column based:\n', cb
    save_data(input_file, './cb.csv', cb, leading_row=lr, leading_column=lc)

    print 'combined:'
    print (rb+cb)/2
    save_data(input_file, './combined.csv', (rb+cb)/2, leading_row=lr, leading_column=lc)


if __name__ == '__main__':
    main()
