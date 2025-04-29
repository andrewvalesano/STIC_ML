def main():

    directory = "results_segment_rois"
    extractor = "uni"
    run_segmentation = "yes"
    extract_tiles = "yes"
    tile_px = 512
    tile_um = "40x"
    stride = 1
    extract_bags = "yes"
    model = "stic_stil" # atypical or stic_stil
    mil_architecture = "attention_mil" # ['mil_fc_mc', 'mm_attention_mil', 'mil_fc', 'transmil', 'clam_sb', 'attention_mil', 'bistro.transformer', 'clam_mb']
    model_stage = "train" # train or eval
    roi_method = "auto" # ignore (weakly supervised) or auto (strongly supervised)
    qc_method = "both" # otsu, blur, or both
    min_tiles = 8
    roi_filter_method = 0.0001
    generate_mosaic = "yes"
    normalization_method = "reinhard" # See options: https://slideflow.dev/norm/#stain-augmentation
    get_top_tiles = "no"
    print_thumbnails = "no"

    slide_location = # removed
    print("\nLoading modules...\n\n")

    import pandas as pd
    import numpy as np
    import gdown
    import transformers
    import torch
    import torchvision
    import slideflow as sf
    import slideflow.model.torch
    from slideflow.model import build_feature_extractor, list_extractors, is_extractor
    from slideflow.mil import mil_config, train_mil, get_mil_tile_predictions, predict_slide, predict_multimodal_mil
    from slideflow.grad import SaliencyMap
    from slideflow.grad.plot_utils import overlay
    from PIL import Image

    print("================================================")
    print("Running with these model parameters:\n")
    print("Directory: " + directory + "\n")
    print("Feature extractor: " + extractor + "\n")
    print("Model: " + model + "\n")
    print("MIL Architecture: " + mil_architecture + "\n")
    print("Model Generation Stage: " + model_stage + "\n")
    print("================================================\n\n")

    P = sf.load_project(directory)

    D = P.dataset(tile_px = tile_px, tile_um = tile_um)
    D.summary()

    if run_segmentation == "yes":
        print("\nRunning the segmentation process...\n\n")
        D.generate_rois(directory + '/roi_segment_output_train/model.pth') # U-Net model for epithelium segmentation

    if extract_tiles == "yes":
        print("\nExtracting tiles...\n\n")
        D.extract_tiles(qc = qc_method, roi_method = roi_method, report = True, stride_div = stride, normalizer = normalization_method, roi_filter_method = roi_filter_method)

    D = D.filter(min_tiles = min_tiles)

    if extract_bags == "yes":
        print("\n\nListing extractors...\n\n")
        print(list_extractors())
        feature_extractor = slideflow.model.build_feature_extractor(extractor, weights = "uni_pytorch_model_weights.bin", resize = 224)
        print("\n\nStarting to generate feature bags...\n\n")
        P.generate_feature_bags(feature_extractor, D, outdir = directory + "/mil/bags_" + extractor + "_" + str(tile_px) + "_" + str(tile_um))

    config = mil_config(model = mil_architecture, lr = 1e-4, batch_size = 32, epochs = 30, fit_one_cycle = True)

    model_dataset = "dataset_" + model
    model_column = "category_" + model

    bags_10x = directory + "/mil/bags_" + extractor + "_" + str(tile_px) + "_10x"
    bags_40x = directory + "/mil/bags_" + extractor + "_" + str(tile_px) + "_40x"

    if model_stage == "train":

        # Train with k-fold cross validation
        training = D.filter({model_dataset: ['train']})
        splits = training.kfold_split(k = 3, labels = model_column)
        for train, test in splits:
            train_mil(config, train_dataset = train, val_dataset = test, outcomes = model_column, bags = bags_40x, outdir = directory + '/mil/model_' + mil_architecture + "_" + model)

        # Train across the entire dataset and evaluate on the holdout data
        eval = D.filter({model_dataset: ['eval']})
        train_mil(config, train_dataset = training, val_dataset = eval,  outcomes = model_column, bags = bags_40x, outdir = directory + '/mil/model_' + mil_architecture + "_" + model, attention_heatmaps = True, interpolation = None, cmap = "inferno")

        # Save the tile predictions and plot top tiles over the slide image
        tile_predictions = get_mil_tile_predictions(weights = directory + "/mil/model_" + mil_architecture + "_" + model + "/00003-" + mil_architecture  +  "-" + model_column  + "/", dataset = eval, bags = bags_40x)
        tile_prediction_file = directory + "/mil/model_" + mil_architecture + "_" + model + "/00003-" + mil_architecture  +  "-" + model_column  + "/" + "/model_" + mil_architecture + "_" + model_column + "_attention_scores.csv"
        tile_predictions.to_csv(tile_prediction_file, index = False)

        grouped = tile_predictions.groupby('slide')
        os.system("mkdir " + directory + "/training_eval_top_attention/")
        for slide, group in grouped:
            top_rows = group.nlargest(10, 'attention')
            coordinates = list(zip(top_rows['loc_x'], top_rows['loc_y']))
            wsi = sf.WSI(slide_location + slide + ".svs", tile_px = tile_px, tile_um = tile_um)
            slide_image = wsi.thumb(coords = coordinates)
            slide_image.save(directory + "/training_eval_top_attention/" + slide + "_top_attention.png")

    if model_stage == "eval":

        eval = D.filter({model_dataset: ['eval']}) 

        P.evaluate_mil(directory + "/mil/model_" + mil_architecture + "_" + model + "/00003-" + mil_architecture  +  "-" + model_column  + "/", 
            outcomes = model_column, dataset = eval, 
            bags = directory + "/mil/bags_" + extractor + "_" + str(tile_px) + "_" + str(tile_um), 
            attention_heatmaps = False, cmap = "inferno", interpolation = "none")

        tile_predictions = get_mil_tile_predictions(weights = directory + "/mil/model_" + mil_architecture + "_" + model + "/00003-" + mil_architecture  +  "-" + model_column  + "/", dataset = eval, bags = directory + "/mil/bags_" + extractor + "_" + str(tile_px) + "_" + str(tile_um))
        tile_predictions.to_csv(directory + "/mil_eval/model_" + mil_architecture + "_" + model_column + "_attention_scores.csv")

    if get_top_tiles == "yes":

        print("\nGetting top tiles and printing out over the H&E...\n\n")
        eval = D.filter({model_dataset: ['eval']})
        tile_predictions = get_mil_tile_predictions(weights = directory + "/mil/model_" + mil_architecture + "_" + model + "/00003-" + mil_architecture  +  "-" + model_column  + "/", dataset = eval, bags = directory + "/mil/bags_" + extractor + "_" + str(tile_px) + "_" + str(tile_um))
        grouped = tile_predictions.groupby('slide')
        os.system("mkdir " + directory + "/eval_mil/eval_top_tiles/")
        for slide, group in grouped:
            #top_row = group.nlargest(1, "attention") # Get top n tiles, regardless of score cutoff
            top_rows = group[group["attention"] > 2.0]
            if not top_rows.empty:
                coordinates = list(zip(top_rows['loc_x'], top_rows['loc_y']))
                wsi = sf.WSI(slide_location + slide + ".svs", tile_px = tile_px, tile_um = tile_um)
                #wsi_tile_df = wsi.get_tile_dataframe()
                #wsi_tile_df.to_csv(directory + "/eval_mil/eval_top_tiles/" + slide + "_tile_df.csv")
                #grid_x, grid_y = lookup_grid(wsi_tile_df, top_row['loc_x'].iloc[0], top_row['loc_y'].iloc[0])
                slide_image = wsi.thumb(coords = coordinates)
                slide_image.save(directory + "/eval_mil/eval_top_tiles/" + slide + "_with_top_tiles.png")

    if generate_mosaic == "yes":
        mosaic_extractor = sf.build_feature_extractor(extractor, weights = "uni_pytorch_model_weights.bin", resize = 224)
        dataset_eval = D.filter({model_dataset: ['train']}) 
        features = sf.DatasetFeatures(mosaic_extractor, dataset = dataset_eval)
        slide_map = features.map_activations()
        mosaic = sf.Mosaic(slide_map, tfrecords = features.tfrecords)
        mosaic.save(directory + "/mosaic_files_mil")
        umap = mosaic.slide_map
        umap.save(directory + "/mosaic_files_mil")
        labels, unique = dataset_eval.labels(model_column)
        umap.label_by_slide(labels)
        umap.save_plot(directory + '/mosaic_files_mil/umap_by_category.png')


import os
import multiprocessing
from multiprocessing import process, freeze_support
import ssl

if __name__ == '__main__':
    freeze_support()
    ssl._create_default_https_context = ssl._create_unverified_context
    main()
