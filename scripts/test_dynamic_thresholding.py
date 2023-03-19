import torch
from dynamic_thresholding import DynamicThresholding


def test_dynthresh():
    # Create a sample instance of the DynamicThresholding class
    dynamic_thresholding = DynamicThresholding()

    # Define input tensors for testing
    x_out = torch.randn(10, 3, 64, 64)
    denoised_uncond = torch.randn(10, 3, 64, 64)
    cond_scale = torch.randn(10, 3, 64, 64)
    conds_list = [torch.randn(10, 3, 64, 64) for _ in range(5)]

    # Call the dynthresh function with the sample input tensors
    try:
        output = dynamic_thresholding.dynthresh(x_out, denoised_uncond, cond_scale, conds_list)
        print("Test passed: No errors encountered.")
    except RuntimeError as e:
        print("Test failed: RuntimeError encountered.")
        print(e)

if __name__ == "__main__":
    test_dynthresh()
