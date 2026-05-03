import numpy as np
import pandas as pd


def dice_to_iou(dice):
    dice = np.asarray(dice, dtype=float)
    if np.any((dice < 0) | (dice > 1)):
        raise ValueError("Dice values must be in [0, 1].")
    return dice / (2.0 - dice)


def iou_to_dice(iou):
    iou = np.asarray(iou, dtype=float)
    if np.any((iou < 0) | (iou > 1)):
        raise ValueError("IoU values must be in [0, 1].")
    return 2.0 * iou / (1.0 + iou)


def convert_and_show(dice_list=None, iou_list=None, decimals=4):
    if (dice_list is None) == (iou_list is None):
        raise ValueError("Provide exactly one of dice_list or iou_list.")

    if dice_list is not None:
        dice = np.asarray(dice_list, dtype=float)
        iou = dice_to_iou(dice)
        df = pd.DataFrame({
            "Dice": np.round(dice, decimals),
            "IoU": np.round(iou, decimals),
        })
    else:
        iou = np.asarray(iou_list, dtype=float)
        dice = iou_to_dice(iou)
        df = pd.DataFrame({
            "IoU": np.round(iou, decimals),
            "Dice": np.round(dice, decimals),
        })

    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    # 示例1：已知 Dice，生成 IoU
    dice_values = [0.886]
    print("From Dice to IoU:")
    convert_and_show(dice_list=dice_values)

    print("\n" + "-" * 40 + "\n")

    # # 示例2：已知 IoU，生成 Dice
    # iou_values = [0.764, 0.729, 0.696, 0.730, 0.765, 0.742, 0.803, 0.733, 0.801, 0.783, 0.776, 0.774, 0.808, 0.829, 0.817, 0.868]
    # print("From IoU to Dice:")
    # convert_and_show(iou_list=iou_values)