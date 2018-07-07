from util.c_matrix import cmatrix, print_cmax, plot_acc_history, plot_loss_history, plot_confusion_matrix, cmatrix_generator, plot_class_acc_history


def present_results_generator(work_dir, model, logs, validation_generator, val_data_count,classes,suffix='def',train_top_epochs=None):
    # present results
    results_file = work_dir + '/results-'+suffix+'.txt'
    cm_image_file = work_dir + '/cm-'+suffix+'.png'
    normalized_cm_image_file = work_dir + '/cm_n-'+suffix+'.png'
    acc_history_image_file = work_dir + '/acc_history-'+suffix+'.png'
    class_place_holder='%cls%'
    class_acc_history_image_file = work_dir + '/acc_history-'+suffix+'-'+class_place_holder+'.png'
    loss_history_image_file = work_dir + '/loss_history-'+suffix+'.png'

    if logs:
        nb_classes = len(validation_generator.class_indices)
        plot_acc_history(logs, acc_history_image_file,train_top_epochs)
        plot_loss_history(logs, loss_history_image_file,train_top_epochs)

        if 'train_per_class' in logs:
            for i in range(nb_classes):
                class_name=validation_generator.class_indices.keys()[validation_generator.class_indices.values().index(i)]
                plot_class_acc_history(i, logs, class_acc_history_image_file.replace(class_place_holder,class_name), vertical_line=train_top_epochs)


    confusion_matrix = cmatrix_generator(model, validation_generator, val_data_count,nb_classes=len(classes))
    validation_result = model.evaluate_generator(validation_generator,
                                                 val_data_count / 1)  # validation batch size = 1
    cm=confusion_matrix
    N = len(cm)

    tp = sum(cm[i][i] for i in range(N))
    fn = sum((sum(cm[i][i + 1:]) for i in range(N)))
    fp = sum(sum(cm[i][:i]) for i in range(N))

    precision = tp * 1.0 / (tp + fp)
    recall = tp * 1.0 / (tp + fn)
    validation_result.extend([precision, recall])

    print_cmax(results_file, confusion_matrix, validation_result)

    plot_confusion_matrix(confusion_matrix, cm_image_file, classes=classes)
    plot_confusion_matrix(confusion_matrix, normalized_cm_image_file, classes=classes, normalize=True)


def present_results(work_dir, model, logs, X_test, Y_test,classes):
    # present results
    results_file = work_dir + '/results.txt'
    cm_image_file = work_dir + '/cm.png'
    normalized_cm_image_file = work_dir + '/cm_n.png'
    acc_history_image_file = work_dir + '/acc_history.png'
    loss_history_image_file = work_dir + '/loss_history.png'

    if logs:
        plot_acc_history(logs, acc_history_image_file)
        plot_loss_history(logs, loss_history_image_file)

    confusion_matrix = cmatrix(model, X_test, Y_test)
    validation_result = model.evaluate(X_test,Y_test,batch_size=1)
    print_cmax(results_file, confusion_matrix, validation_result)

    plot_confusion_matrix(confusion_matrix, cm_image_file, classes=classes)
    plot_confusion_matrix(confusion_matrix, normalized_cm_image_file, classes=classes,normalize=True)