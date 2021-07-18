import shutil
from pathlib import Path

from matplotlib import pyplot as plt


def make_combined_plots_horizontal(plots, replace_string='50_frames_pred'):
    plt.figure()
    fig, axs = plt.subplots(2, 1, figsize=(6, 5), dpi=300, sharex=True)

    plt.rcParams.update({'font.size': 9})
    num_epochs_after_max = 100

    palette = {'MISLNet': ('tab:orange', '-'),
               'MISLNet - Constrained': ('tab:orange', '--'),
               'MobileNet': ('tab:green', '-'),
               'MobileNet - Constrained': ('tab:green', '--'),
               'EfficientNet': ('tab:blue', '-'),
               'EfficientNet - Constrained': ('tab:blue', '--')
               }

    for plot_name in plots:
        if plots[plot_name].exists():
            color = palette[plot_name][0]
            style = palette[plot_name][1]
            with open(plots[plot_name], 'r') as f:
                lines = sorted(f.readlines()[2:])
            test_loss = [float(x.split(',')[3]) for x in lines]
            test_acc = [float(x.split(',')[1]) for x in lines]

            max_elem = max(zip(test_acc, [-x for x in test_loss]))
            max_elem = (max_elem[0], -max_elem[1])
            max_acc_indices = set([i for i, x in enumerate(test_acc) if x == max_elem[0]])
            min_loss_indices = set([i for i, x in enumerate(test_loss) if x == max_elem[1]])
            index = min(max_acc_indices.intersection(min_loss_indices))

            model_path = list(plots[plot_name].parent.parent.parent.glob(f'*{str(index + 1).zfill(5)}.h5'))[0]
            tmp = list(model_path.parts)[:-1]
            tmp[-4] = replace_string
            dest_path = Path('/'.join(tmp))
            dest_path.mkdir(exist_ok=True, parents=True)
            shutil.copy(model_path, dest_path)

            test_loss = test_loss[:index + num_epochs_after_max]
            test_acc = test_acc[:index + num_epochs_after_max]
            epochs = list(range(1, len(test_loss) + 1))

            axs[0].plot(epochs, test_acc, label=plot_name, alpha=0.7, color=color, linestyle=style)
            color = axs[0].lines[-1].get_color()
            axs[0].scatter(index + 1, test_acc[index], color=color, s=12)

    for plot_name in plots:
        if plots[plot_name].exists():
            color = palette[plot_name][0]
            style = palette[plot_name][1]
            with open(plots[plot_name], 'r') as f:
                lines = sorted(f.readlines()[2:])
            test_loss = [float(x.split(',')[3]) for x in lines]
            test_acc = [float(x.split(',')[1]) for x in lines]

            max_elem = max(zip(test_acc, [-x for x in test_loss]))
            max_elem = (max_elem[0], -max_elem[1])
            max_acc_indices = set([i for i, x in enumerate(test_acc) if x == max_elem[0]])
            min_loss_indices = set([i for i, x in enumerate(test_loss) if x == max_elem[1]])
            index = min(max_acc_indices.intersection(min_loss_indices))

            test_loss = test_loss[:index + num_epochs_after_max]
            test_acc = test_acc[:index + num_epochs_after_max]
            epochs = list(range(1, len(test_loss) + 1))

            axs[1].plot(epochs, test_loss, label=plot_name, alpha=0.7, color=color, linestyle=style)
            color = axs[1].lines[-1].get_color()
            axs[1].scatter(index + 1, test_loss[index], color=color, s=12)

    # plt.suptitle('Epoch-wise accuracy and loss on the test set')

    # axs[0].set_xlabel('epochs')
    axs[0].set_ylabel('Validation  accuracy')
    axs[0].set_ylim([0.20, 0.70])
    axs[0].set_xlim([0, 20])

    axs[1].set_xlabel('epochs')
    axs[1].set_ylabel('Validation  loss')
    axs[1].set_ylim([0.20, 0.37])
    axs[1].set_xlim([0, 21])

    plt.subplots_adjust(wspace=0.05, hspace=0.08)

    # plt.xlabel('epochs')

    # axes = plt.gca()
    # plt.ylabel('accuracy')
    # axes.set_ylim([0.40, 0.74])

    # plt.title('Epoch-wise loss on test data')
    # plt.ylabel('loss')
    # axes.set_ylim([0.218, 0.31])

    # axs[1].legend(loc='upper right')
    axs[0].grid(linewidth=0.2, axis='x', linestyle='--')
    axs[1].grid(linewidth=0.2, axis='x', linestyle='--')

    plt.legend(bbox_to_anchor=(0.98, 1.45), loc='upper right', borderaxespad=0., framealpha=0.95)
    # plt.tight_layout()

    plt.show()
    plt.cla()
    plt.clf()
    plt.close()


if __name__ == '__main__':
    # plot_data = {
    #     'MISLNet': Path(
    #         r'/scratch/p288722/runtime_data/scd_videos_tf/50_frames/misl_net_2/models/ConvNet/'
    #         r'predictions_50_frames_bal_val/videos/V_prediction_stats.csv'),
    #     'MISLNet - Constrained': Path(
    #         r'/scratch/p288722/runtime_data/scd_videos_tf/50_frames/misl_net_2/models/ConstNet/'
    #         r'predictions_50_frames_bal_val/videos/V_prediction_stats.csv'),
    #     'EfficientNet': Path(
    #         r'/scratch/p288722/runtime_data/scd_videos_tf/50_frames/efficient_net/models/ConvNet/'
    #         r'predictions_50_frames_bal_val/videos/V_prediction_stats.csv'),
    #     'EfficientNet - Constrained': Path(
    #         r'/scratch/p288722/runtime_data/scd_videos_tf/50_frames/efficient_net/models/ConstNet/'
    #         r'predictions_50_frames_bal_val/videos/V_prediction_stats.csv'),
    #     'MobileNet': Path(
    #         r'/scratch/p288722/runtime_data/scd_videos_tf/50_frames/mobile_net_1/models/ConvNet/'
    #         r'predictions_50_frames_bal_val/videos/V_prediction_stats.csv'),
    #     'MobileNet - Constrained': Path(
    #         r'/scratch/p288722/runtime_data/scd_videos_tf/50_frames/mobile_net_2/models/ConstNet/'
    #         r'predictions_50_frames_bal_val/videos/V_prediction_stats.csv'),
    # }
    # make_combined_plots_horizontal(plot_data)

    plot_data = {
        'MobileNet - Constrained': Path(
            r'/scratch/p288722/runtime_data/scd-videos/no_frame_selection/50_frames_8d_64/mobile_net/models/'
            r'h0_lab_ConstNet_derrick/predictions_50_frames_val/videos/V_prediction_stats.csv'),
    }
    make_combined_plots_horizontal(plot_data, replace_string='50_frames_8d_64_pred')

    plot_data = {
        'MobileNet - Constrained': Path(
            r'/scratch/p288722/runtime_data/scd-videos/no_frame_selection/50_frames_8d_64/mobile_net/models/'
            r'h0_lab_ConvNet/predictions_50_frames_val/videos/V_prediction_stats.csv'),
    }
    make_combined_plots_horizontal(plot_data, replace_string='50_frames_8d_64_pred')

    plot_data = {
        'MobileNet - Constrained': Path(
            r'/scratch/p288722/runtime_data/scd-videos/i_frames/all_frames_8d_64/mobile_net/models/'
            r'h0_lab_ConstNet_derrick/predictions_all_frames_val/videos/V_prediction_stats.csv'),
    }
    make_combined_plots_horizontal(plot_data, replace_string='all_frames_8d_64_pred')

    plot_data = {
        'MobileNet - Constrained': Path(
            r'/scratch/p288722/runtime_data/scd-videos/i_frames/all_frames_8d_64/mobile_net/models/'
            r'h0_lab_ConvNet/predictions_all_frames_val/videos/V_prediction_stats.csv'),
    }
    make_combined_plots_horizontal(plot_data, replace_string='all_frames_8d_64_pred')
