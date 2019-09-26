'''
Created on Dec 18, 2018
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import tensorflow as tf
from utility.helper import *
from utility.batch_test import *
from time import time

from KGAT import KGAT

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


if __name__ == '__main__':
    # get argument settings.
    #tf.random.set_seed(2019)
    tf.set_random_seed(2019)
    np.random.seed(2019)
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    """
    *********************************************************
    Load Data from data_generator function.
    """
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['n_relations'] = data_generator.n_relations
    config['n_entities'] = data_generator.n_entities

    if args.model_type in ['kgat', 'cfkg']:
        "Load the laplacian matrix."
        config['A_in'] = sum(data_generator.lap_list)

        "Load the KG triplets."
        config['all_h_list'] = data_generator.all_h_list
        config['all_r_list'] = data_generator.all_r_list
        config['all_t_list'] = data_generator.all_t_list
        config['all_v_list'] = data_generator.all_v_list
    print("Data statistics:\n\tn_users:{}, n_items:{}, n_relations:{}, n_entities:{}".format(
        data_generator.n_users, data_generator.n_items, data_generator.n_relations,
        data_generator.n_entities ) )

    t0 = time()
    model = KGAT(data_config=config, args=args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    """
    *********************************************************
    Reload the model parameters to fine tune.
    """
    sess.run(tf.global_variables_initializer())
    cur_best_pre_0 = 0.
    print('without pretraining.')

    """
    *********************************************************
    Train.
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False

    for epoch in range(args.epoch):
        t1 = time()
        loss, base_loss, kge_loss, reg_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        """
        *********************************************************
        Alternative Training for KGAT:
        ... phase 1: to train the recommender.
        """
        for idx in range(n_batch):
            btime= time()

            batch_data = data_generator.generate_train_batch()
            feed_dict = data_generator.generate_train_feed_dict(model, batch_data)

            _, batch_loss, batch_base_loss, batch_kge_loss, batch_reg_loss = model.train(sess, feed_dict=feed_dict)

            loss += batch_loss
            base_loss += batch_base_loss
            kge_loss += batch_kge_loss
            reg_loss += batch_reg_loss

        if np.isnan(loss) == True:
            print('ERROR: loss@phase1 is nan.')
            sys.exit()

        """
        *********************************************************
        Alternative Training for KGAT:
        ... phase 2: to train the KGE method & update the attentive Laplacian matrix.
        """
        if args.model_type in ['kgat']:

            n_A_batch = len(data_generator.all_h_list) // args.batch_size_kg + 1

            if args.use_kge is True:
                # using KGE method (knowledge graph embedding).
                for idx in range(n_A_batch):
                    btime = time()
                    A_batch_data = data_generator.generate_train_A_batch()
                    feed_dict = data_generator.generate_train_A_feed_dict(model, A_batch_data)

                    _, batch_loss, batch_kge_loss, batch_reg_loss = model.train_A(sess, feed_dict=feed_dict)

                    loss += batch_loss
                    kge_loss += batch_kge_loss
                    reg_loss += batch_reg_loss

            if args.use_att is True:
                # updating attentive laplacian matrix.
                model.update_attentive_A(sess)

        if np.isnan(loss) == True:
            print('ERROR: loss@phase2 is nan.')
            sys.exit()

        show_step = 10
        if (epoch + 1) % show_step != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
                    epoch, time() - t1, loss, base_loss, kge_loss, reg_loss)
                print(perf_str)
            continue

        """
        *********************************************************
        Test.
        """
        t2 = time()
        users_to_test = list(data_generator.test_user_dict.keys())

        ret = test(sess, model, users_to_test, drop_flag=False, batch_test_flag=batch_test_flag)

        """
        *********************************************************
        Performance logging.
        """
        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, base_loss, kge_loss, reg_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.


    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write('embed_size=%d, lr=%.4f, layer_size=%s, node_dropout=%s, mess_dropout=%s, regs=%s, loss_type=%s, n_memory=%d,'
            'item_update_mode=%s, using_all_hops=%s, adj_type=%s, use_att=%s, use_kge=%s\n\t%s\n'
            % (args.embed_size, args.lr, args.layer_size, args.node_dropout, args.mess_dropout, args.regs, args.loss_type, args.n_memory,
               args.item_update_mode, args.using_all_hops, args.adj_type, args.use_att, args.use_kge, final_perf))
    f.close()








