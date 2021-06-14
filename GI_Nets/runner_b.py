from utils.utils import results_to_latex

if __name__ == "__main__":
    net_name = "unet_05_10_2021__00_16_30_tanh_ssim"
    results_path = "/home/ksp/Thesis/src/Thesis/GI_Nets/DeepShadingBased/results/ssim_kernel/{}/results_secondary_test_set.json".format(net_name)
    print(results_to_latex(results_path))