

class Tester:
    def __init__(self):
        self.test_cases = []
        self.results = []

    def __load_weight(self):
        weight = torch.load(self.args.weight_path)
        if self.args.weight_path[-1] == 'r':
            weight = weight['state_dict']
        else:
            weight = weight
        self.model.load_state_dict(weight)
        self.warm_epoch = -1

    def test():
        self.model.eval()
        self.mIoU.reset()
        self.PD_FA.reset()
        losses = AverageMeter()
        tbar = tqdm(self.val_loader)
        tag = False
        with torch.no_grad():
            for i, (data, mask) in enumerate(tbar):
                
                mask = mask.to(self.device)
                data = data.to(self.device)

                if epoch>self.warm_epoch:
                    tag = True

                loss = 0
                _, pred = self.model(data, tag)
                self.mIoU.update(pred, mask)
                self.PD_FA.update(pred, mask)
                self.ROC.update(pred, mask)

                # 保存每张输出图像
                if self.mode == 'test':
                    data_image = data.cpu().detach()
                    mask_image = mask.cpu().detach()
                    pred_image = pred.cpu().detach()
                    save_image(data_image, os.path.join('/home/lsfwtt/UAV/reference/HDNet/test', f'{i}_data.png'))
                    save_image(mask_image, os.path.join('/home/lsfwtt/UAV/reference/HDNet/test', f'{i}_mask.png'))
                    save_image(pred_image, os.path.join('/home/lsfwtt/UAV/reference/HDNet/test', f'{i}_pred.png'))

                _, mean_IoU = self.mIoU.get()
                tbar.set_description('Epoch %d, IoU %.4f, loss %.4f' % (epoch, mean_IoU, losses.avg))
            FA, PD = self.PD_FA.get(len(self.val_loader))
            _, mean_IoU = self.mIoU.get()
            ture_positive_rate, false_positive_rate, _, _ = self.ROC.get()

            if self.mode == 'train':
                with open(os.path.join(self.save_folder, 'log.txt'), 'a') as f:
                        f.write('{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n' .
                            format(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())), 
                                epoch, mean_IoU, PD[0], FA[0] * 1000000))
                if mean_IoU > self.best_iou:
                    self.best_iou = mean_IoU
                
                    torch.save(self.model.state_dict(), self.save_folder+'/weight.pkl')
                    with open(os.path.join(self.save_folder, 'metric.log'), 'a') as f:
                        f.write('{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n' .
                            format(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())), 
                                epoch, self.best_iou, PD[0], FA[0] * 1000000))
                        
                all_states = {"net":self.model.state_dict(), "optimizer":self.optimizer.state_dict(), "epoch": epoch, "iou":self.best_iou}
                torch.save(all_states, self.save_folder+'/checkpoint.pkl')
            elif self.mode == 'test':
                print('mIoU: '+str(mean_IoU)+'\n')
                print('Pd: '+str(PD[0])+'\n')
                print('Fa: '+str(FA[0]*1000000)+'\n')