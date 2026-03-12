
import matplotlib.pyplot as plt
def draw_curve_train(path, iter, train_loss):
    fig = plt.figure(figsize=(12, 8))  # 调整画布大小
    ax1 = fig.add_subplot(title="train_loss")


    # 绘制 train_loss 曲线
    ax1.plot(iter, train_loss, 'bo-', label='train_loss: {:.3f}'.format(train_loss[-1]))

    ax1.legend()

    # 保存图像
    fig.savefig(path)
    plt.close(fig)



def draw_curve_test(path, step, test_mse, test_mae, test_ot, test_MMD, test_FDE, test_DTW, test_coll):
    fig = plt.figure(figsize=(12, 8))  # 调整画布大小
    ax1 = fig.add_subplot(331, title="Test MSE")
    ax2 = fig.add_subplot(332, title="Test MAE")
    ax3 = fig.add_subplot(333, title="Test OT")
    ax4 = fig.add_subplot(334, title="Test MMD")
    ax5 = fig.add_subplot(335, title="Test FDE")
    ax6 = fig.add_subplot(336, title="Test DTW")
    ax7 = fig.add_subplot(337, title="Test Col")

    # 绘制 test_fvd 曲线
    ax1.plot(step, test_mse, 'bo-', label='test_mse: {:.4f}'.format(test_mse[-1]))
    ax1.legend()

    # 绘制 test_ssim 曲线
    ax2.plot(step, test_mae, 'ro-', label='test_mae: {:.4f}'.format(test_mae[-1]))
    ax2.legend()

    # 绘制 test_psnr 曲线
    ax3.plot(step, test_ot, 'go-', label='test_ot: {:.4f}'.format(test_ot[-1]))
    ax3.legend()

    # 绘制 test_lpips 曲线
    ax4.plot(step, test_MMD, 'mo-', label='test_MMD: {:.4f}'.format(test_MMD[-1]))
    ax4.legend()

    ax5.plot(step, test_FDE, 'mo-', label='test_FDE: {:.4f}'.format(test_FDE[-1]))
    ax5.legend()

    ax6.plot(step, test_DTW, 'mo-', label='test_DTW: {:.4f}'.format(test_DTW[-1]))
    ax6.legend()


    ax7.plot(step, test_coll, 'mo-', label="Test Collision: {:.4f}".format(test_coll[-1]))
    ax7.legend()

    # 调整布局以防止重叠
    plt.tight_layout()

    # 保存图像
    fig.savefig(path)
    plt.close(fig)