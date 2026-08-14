from logging import getLogger

import torch
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from VRPModel import VRPModel as Model
from test import main_test
from VRPEnv import VRPEnv as Env
from utils.utils import *

class VRPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # MTP specific parameters
        self.mtp_weight = model_params.get('mtp_weight', 0.5)
        self.warmup_epochs = trainer_params.get('mtp_warmup_epochs', 0)

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()
        random_seed = 22
        torch.manual_seed(random_seed)
        # cuda
        USE_CUDA = self.trainer_params['use_cuda'] # True
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num'] # 0
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params) # # {'problem_size': 100, 'pomo_size': 100}

        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):

        self.time_estimator.reset(self.start_epoch)

        self.env.load_raw_data(self.trainer_params['train_episodes'] )

        save_gap = []
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')
            self.env.shuffle_data()
            # Train with MTP
            train_score, train_student_score, train_loss, mtp_losses = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_student_score', epoch, train_student_score)
            self.result_log.append('train_loss', epoch, train_loss)
            
            # Log MTP losses
            for mtp_key, mtp_value in mtp_losses.items():
                self.result_log.append(f'train_{mtp_key}', epoch, mtp_value)
            
            # LR Decay
            self.scheduler.step()

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],self.result_log, labels=['train_loss'])

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

                score_optimal, score_student ,gap = main_test(epoch,self.result_folder, use_RRC = False,
                                                              cuda_device_num=self.trainer_params['cuda_device_num'])

                save_gap.append([score_optimal, score_student,gap])
                np.savetxt(self.result_folder+'/gap.txt',save_gap,delimiter=',',fmt='%s')

            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],self.result_log, labels=['train_loss'])

            if all_done:
                self.logger.info(" *** MTP Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        loss_AM = AverageMeter()
        mtp_loss_AMs = {}

        train_num_episode = self.trainer_params['train_episodes'] # 100000
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, score_student_mean, avg_loss, avg_mtp_losses = self._train_one_batch(episode, batch_size, epoch)

            score_AM.update(avg_score, batch_size)
            score_student_AM.update(score_student_mean, batch_size)
            loss_AM.update(avg_loss, batch_size)
            
            # Update MTP loss averages
            for mtp_key, mtp_value in avg_mtp_losses.items():
                if mtp_key not in mtp_loss_AMs:
                    mtp_loss_AMs[mtp_key] = AverageMeter()
                mtp_loss_AMs[mtp_key].update(mtp_value, batch_size)

            episode += batch_size

            loop_cnt += 1
            
            # Create log string for MTP losses
            # mtp_log_str = ", ".join([f"{k}: {v.avg:.4f}" for k, v in mtp_loss_AMs.items()]) if mtp_loss_AMs else "No MTP"
            
            # # Log every 100 batches to monitor training
            # if loop_cnt % 1 == 0:
            #     self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f}, Score_studetnt: {:.4f},  Loss: {:.4f}, {}'
            #                      .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode, score_AM.avg, score_student_AM.avg, loss_AM.avg, mtp_log_str))

        # Final epoch log
        final_mtp_losses = {k: v.avg for k, v in mtp_loss_AMs.items()}
        final_mtp_str = ", ".join([f"{k}: {v:.4f}" for k, v in final_mtp_losses.items()]) if final_mtp_losses else "No MTP"

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f}, Score_studetnt: {:.4f}, Loss: {:.4f}, {}'
                         .format(epoch, 100. * episode / train_num_episode,score_AM.avg, score_student_AM.avg, loss_AM.avg, final_mtp_str))

        return score_AM.avg, score_student_AM.avg, loss_AM.avg, final_mtp_losses

    def _train_one_batch(self, episode, batch_size, epoch):

        self.model.train() # training mode

        self.env.load_problems(episode,batch_size) # regenerate problems

        reset_state, _, _ = self.env.reset(self.env_params['mode'])

        loss_list = []
        mtp_losses_batch = {}

        state, reward,reward_student, done = self.env.pre_step()

        current_step=0

        while not done:

            if current_step ==0:
                # print('current_step', current_step)
                selected_teacher = self.env.solution[:, 0, 0]
                selected_flag_teacher = self.env.solution[:, 0, 1]
                selected_student = selected_teacher
                selected_flag_student = selected_flag_teacher
                loss_mean = torch.tensor(0)
                mtp_losses = {}

            else:
                # Get predictions and MTP losses
                result = self.model(state, self.env.selected_node_list, self.env.solution, current_step,
                                   raw_data_capacity=self.env.raw_data_capacity)
                
                if len(result) == 6:  # Training mode with MTP losses
                    loss_node, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student, mtp_losses = result
                else:  # Test mode or no MTP
                    loss_node, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student = result
                    mtp_losses = {}

                # Compute standard next-token prediction loss
                ntp_loss = loss_node
                
                # Compute total MTP loss with validation
                total_mtp_loss = torch.tensor(0.0, device=loss_node.device if torch.is_tensor(loss_node) else torch.device('cpu'))
                valid_mtp_count = 0
                
                for mtp_key, mtp_loss in mtp_losses.items():
                    # Only include valid, meaningful MTP losses
                    if (torch.is_tensor(mtp_loss) and 
                        not torch.isnan(mtp_loss) and 
                        not torch.isinf(mtp_loss) and 
                        mtp_loss > 0):
                        
                        total_mtp_loss += mtp_loss
                        valid_mtp_count += 1
                        
                        # Accumulate batch-wise MTP losses
                        if mtp_key not in mtp_losses_batch:
                            mtp_losses_batch[mtp_key] = []
                        mtp_losses_batch[mtp_key].append(mtp_loss.item())
                
                # Adaptive MTP weighting (start with lower weight, gradually increase)
                current_mtp_weight = self._get_adaptive_mtp_weight(epoch)
                
                # Only add MTP loss if we have valid losses
                if valid_mtp_count > 0:
                    total_mtp_loss = total_mtp_loss / valid_mtp_count
                    total_loss = ntp_loss + current_mtp_weight * total_mtp_loss
                else:
                    total_loss = ntp_loss
                
                loss_mean = total_loss

                self.model.zero_grad()
                loss_mean.backward()

                self.optimizer.step()

            current_step+=1
            state, reward, reward_student, done = self.env.step(selected_teacher, selected_student,selected_flag_teacher,selected_flag_student)  # update selected_teacher list and mask

            loss_list.append(loss_mean)

        loss_mean = torch.tensor(loss_list).mean()
        
        # Average MTP losses for this batch
        avg_mtp_losses = {}
        for mtp_key, mtp_values in mtp_losses_batch.items():
            avg_mtp_losses[mtp_key] = sum(mtp_values) / len(mtp_values) if mtp_values else 0.0

        return 0, 0, loss_mean.item(), avg_mtp_losses

    def _get_adaptive_mtp_weight(self, epoch):
        """
        Adaptive MTP weight that starts small and gradually increases
        This prevents MTP from overwhelming early training
        """
        if epoch < self.warmup_epochs:
            # Linear warm-up from 0 to mtp_weight
            return self.mtp_weight * (epoch / (self.warmup_epochs*3))
        else:
            return self.mtp_weight
