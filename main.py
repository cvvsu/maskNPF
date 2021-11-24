# -*- coding: utf-8 -*-


from model import NPFDetection

from options import get_args


if __name__=='__main__':
    opt, message = get_args()
    print(message)

    NPF = NPFDetection(opt)

    #############################
    # draw NPF images
    #############################
    #NPF.draw_one_day_images()
    #NPF.draw_two_day_images()


    #############################
    # Get masks for NPF images
    #############################
    # NPF.detect_one_day_masks()
    # NPF.detect_two_day_masks()

    #############################
    # Visualize and select masks
    #############################
    # NPF.visualize_masks()

    #####################################
    # Save the start-end times and GRs
    #####################################
    NPF.save_SE_GR()
