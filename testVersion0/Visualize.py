#Setting Steps
total_train_step=0
total_test_step=0

#Rounds for training
epoch=20

#data visualize: TensorBoard
writer=SummaryWriter("./logs_cifar")

for i in range(epoch):
    print("-----------------The {} round start-------------------".format(i+1))

    #test result total of every epoch
    final_predict=[]
    final_true=[]
    #start training
    for j,data in enumerate(train_loader):
        imgs,targets=data
        imgs,targets=imgs.to(device),targets.to(device)
        # imgs=imgs.reshape(20,3,74,74)  #batch,channel,img.height, img.weight
        outputs=model(imgs)
        # print(f"outputs: {outputs}")
        # print(f"targets: {targets}")

        #Add class weight to loss function
        loss=loss_func(outputs,targets.long())
        # print(f"loss={loss}")

        #optimizer
        optimizer_train.zero_grad()
        loss.backward()
        optimizer_train.step()
        total_train_step=total_train_step+1
        if total_train_step%25==0:
            print("Train step {}, Loss: {}".format(total_train_step,loss))
            writer.add_scalar("train_loss",loss,total_train_step)

    #start testing
    # Create model
    target_name=[str(i) for i in test_lst]
    # print(f"target_name= {target_name}")
    with torch.no_grad():    #ensure this data would not be optimized
        for l,data in enumerate(test_loader):
            imgs,targets=data
            imgs,targets=imgs.to(device),targets.to(device)

            outputs=model(imgs)
            # outputs = cifar_train(imgs)

            loss=loss_func(outputs,targets.long())
            outputs_np=outputs.numpy()
            #Decide the prediction of outputs --[predict probability of each class]
            x_predict=np.argmax(outputs_np,axis=1)
            x_predict=x_predict.tolist()
            #Get the true labels
            targets_np=targets.numpy()
            targets_np=targets_np
            targets_ls=targets_np.tolist()
            targets_ls=list(map(int,targets_ls[:]))
            y_true = list()
            for q in targets_ls:
                t=test_lst.index(q)
                y_true.append(t)

            #Sum up all batches of label lists
            final_true.extend(y_true)
            final_predict.extend(x_predict)

    print(f'final predict label list= {final_predict}')
    print(f'final true label list= {final_true}')
    #report table fonction
    total_test_step = total_test_step + 1
    report= classification_report(final_true, final_predict)
    report_dict = classification_report(final_true, final_predict,output_dict=True)
    print(report)
    print(report_dict)
    #record result in tensorboard
    writer.add_scalar("0_precision", report_dict['0']['precision'] , total_test_step)
    writer.add_scalar("1_precision", report_dict['1']['precision'], total_test_step)
    writer.add_scalar("0_recall", report_dict['0']['recall'], total_test_step)
    writer.add_scalar("1_recall", report_dict['1']['recall'], total_test_step)
    writer.add_scalar("0_f1-score", report_dict['0']['f1-score'], total_test_step)
    writer.add_scalar("1_f1-score", report_dict['1']['f1-score'], total_test_step)
    writer.add_scalar("total-accuracy", report_dict[], total_test_step)

    # save trained model
    torch.save(model,"model_{}.pth".format(i))
    print("Model saved...")

writer.close()
