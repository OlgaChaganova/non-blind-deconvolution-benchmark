-- SQLite
-- Select all data stored in the database
SELECT *
FROM full_dataset_all_blur_3;

---------------------------------- DATASET EXPLORATION ----------------------------------

-- Select blur types
SELECT blur_type, COUNT(DISTINCT kernel)
FROM full_dataset_all_blur_3
GROUP BY blur_type;


-- Select images
SELECT image_dataset, COUNT(DISTINCT image)
FROM full_dataset_all_blur_3
GROUP BY image_dataset;


SELECT model, COUNT(image)
FROM full_dataset_all_blur_3
GROUP BY model;

---------------------------------- MODEL BENCHMARKING ----------------------------------

-- Select mean metrics for different model types
SELECT load_extension('edu/non-blind-deconvolution-benchmark/sqlean/stats.so');  -- IMPORTANT: you should change this path to yours
SELECT discretization, blur_type, noised, model, MEDIAN(psnr), AVG(psnr), STDDEV(psnr) as 'STD(psnr)', MEDIAN(ssim), AVG(ssim), STDDEV(ssim) as 'STD(ssim)'
FROM full_dataset_all_blur_3
GROUP BY discretization, blur_type, noised, model;

-- Select mean metrics only for Wiener filter
SELECT load_extension('edu/non-blind-deconvolution-benchmark/sqlean/stats.so');  -- IMPORTANT: you should change this path to yours
SELECT discretization, blur_type, noised, model, MEDIAN(psnr), AVG(psnr), STDDEV(psnr) as 'STD(psnr)', MEDIAN(ssim), AVG(ssim), STDDEV(ssim) as 'STD(ssim)'
FROM full_dataset_all_blur_3
WHERE model IN ('wiener_nonblind_noise', 'wiener_blind_noise')
GROUP BY discretization, blur_type, noised, model;


-- Select mean metrics for different model types for all blur types (not quite correct, because not all models were trained on gauss_blur or eye_blur)
SELECT load_extension('edu/non-blind-deconvolution-benchmark/sqlean/stats.so');
SELECT noised, model, MEDIAN(psnr), AVG(psnr), STDDEV(psnr) as 'STD(psnr)', MEDIAN(ssim), AVG(ssim), STDDEV(ssim) as 'STD(ssim)'
FROM full_dataset_all_blur_3
GROUP BY noised, model;


--motion blur (type of blur which was used to training with all models)
SELECT load_extension('edu/non-blind-deconvolution-benchmark/sqlean/stats.so');
SELECT discretization, noised, model, MEDIAN(psnr), AVG(psnr), STDDEV(psnr) as 'STD(psnr)', MEDIAN(ssim), AVG(ssim), STDDEV(ssim) as 'STD(ssim)'
FROM full_dataset_all_blur_3
WHERE blur_type == 'motion_blur'
GROUP BY discretization, noised, model;


----------------------------- float -----------------------------

-- Select mean metrics for different model types and FLOAT dicretization 
-- (not quite correct, because not all models were tested on gauss_blur or eye_blur)
SELECT load_extension('edu/non-blind-deconvolution-benchmark/sqlean/stats.so');
SELECT noised, model, MEDIAN(psnr), AVG(psnr), STDDEV(psnr) as 'STD(psnr)', MEDIAN(ssim), AVG(ssim), STDDEV(ssim) as 'STD(ssim)'
FROM full_dataset_all_blur_3
WHERE discretization == 'float'
GROUP BY noised, model;


-- Select mean metrics for different model types + FLOAT dicretization + MOTION_BLUR
SELECT load_extension('edu/non-blind-deconvolution-benchmark/sqlean/stats.so');
SELECT noised, model, MEDIAN(psnr), AVG(psnr), STDDEV(psnr) as 'STD(psnr)', MEDIAN(ssim), AVG(ssim), STDDEV(ssim) as 'STD(ssim)'
FROM full_dataset_all_blur_3
WHERE discretization == 'float' and blur_type == 'motion_blur'
GROUP BY noised, model;


----------------------------- srgb_8bit -----------------------------

-- Select mean metrics for different model types and UINT8 dicretization
-- (not quite correct, because not all models were tested on gauss_blur or eye_blur)
SELECT load_extension('edu/non-blind-deconvolution-benchmark/sqlean/stats.so');
SELECT noised, model, MEDIAN(psnr), AVG(psnr), STDDEV(psnr) as 'STD(psnr)', MEDIAN(ssim), AVG(ssim), STDDEV(ssim) as 'STD(ssim)'
FROM full_dataset_all_blur_3
WHERE discretization == 'srgb_8bit'
GROUP BY noised, model;


-- Select mean metrics for different model types | FLOAT dicretization | MOTION_BLUR
SELECT load_extension('edu/non-blind-deconvolution-benchmark/sqlean/stats.so');
SELECT noised, model, MEDIAN(psnr), AVG(psnr), STDDEV(psnr) as 'STD(psnr)', MEDIAN(ssim), AVG(ssim), STDDEV(ssim) as 'STD(ssim)'
FROM full_dataset_all_blur_3
WHERE discretization == 'srgb_8bit' and blur_type == 'motion_blur'
GROUP BY noised, model;


----------------------------- linrgb_16bit -----------------------------

-- Select mean metrics for different model types | FLOAT dicretization | MOTION_BLUR
SELECT load_extension('edu/non-blind-deconvolution-benchmark/sqlean/stats.so');
SELECT noised, model, MEDIAN(psnr), AVG(psnr), STDDEV(psnr) as 'STD(psnr)', MEDIAN(ssim), AVG(ssim), STDDEV(ssim) as 'STD(ssim)'
FROM full_dataset_all_blur_3
WHERE discretization == 'linrgb_16bit' and blur_type == 'motion_blur'
GROUP BY noised, model;


---------------------------------- WIENER ----------------------------------

----------------------------- worst restoration -----------------------------

-- Select the WORST restoration examples for WIENER
SELECT blur_type, noised, kernel, image, ssim, psnr
FROM full_dataset_all_blur_3
WHERE discretization == 'float' and model == 'wiener_nonblind_noise' and ssim <= 0.8 
ORDER BY psnr, ssim
LIMIT 20;

-- Select the WORST restoration examples for WIENER
SELECT kernel, image, ssim, psnr
FROM full_dataset_all_blur_3
WHERE noised == 0 AND discretization == 'float' and model == 'wiener_nonblind_noise' and blur_type == 'motion_blur'
ORDER BY psnr, ssim
LIMIT 20;


SELECT kernel, image, ssim, psnr
FROM full_dataset_all_blur_3
WHERE noised == 1 AND discretization == 'float' and model == 'wiener_nonblind_noise' and blur_type == 'motion_blur'
ORDER BY psnr, ssim
LIMIT 20;


-- Select the WORST restoration examples for KERUNC
SELECT kernel, image, ssim, psnr
FROM full_dataset_all_blur_3
WHERE noised == 1 AND discretization == 'float' and model == 'kerunc' and blur_type == 'motion_blur'
ORDER BY psnr, ssim
LIMIT 20;


----------------------------- best restoration -----------------------------

-- float
SELECT blur_type, kernel, image, psnr, ssim
FROM full_dataset_all_blur_3
WHERE noised == 0 AND discretization == 'float' and model == 'wiener_nonblind_noise'
ORDER BY psnr DESC, ssim DESC
LIMIT 20;

-- srgb_8bit
SELECT blur_type, kernel, image, psnr, ssim
FROM full_dataset_all_blur_3
WHERE noised == 0 AND discretization == 'srgb_8bit' and model == 'wiener_nonblind_noise'
ORDER BY psnr DESC, ssim DESC
LIMIT 20;

-- linrgb_16bit
SELECT blur_type, kernel, image, psnr, ssim
FROM full_dataset_all_blur_3
WHERE noised == 0 AND discretization == 'linrgb_16bit' and model == 'wiener_nonblind_noise'
ORDER BY psnr DESC, ssim DESC
LIMIT 20;


-- Select the BEST restoration examples for WIENER with GAUSS BLUR
SELECT blur_type, kernel, image, psnr, ssim
FROM full_dataset_all_blur_3
WHERE noised == 0 AND discretization == 'float' and model == 'wiener_nonblind_noise' and blur_type == 'gauss_blur'
ORDER BY psnr DESC, ssim DESC
LIMIT 20;


-- Select the BEST restoration examples for WIENER with MOTION BLUR
SELECT blur_type, kernel, image, psnr, ssim
FROM full_dataset_all_blur_3
WHERE noised == 0 AND discretization == 'float' and model == 'wiener_nonblind_noise' and blur_type == 'motion_blur'
ORDER BY psnr DESC, ssim DESC
LIMIT 20;


-- Select the BEST restoration examples for WIENER with MOTION BLUR
SELECT blur_type, kernel, image, psnr, ssim
FROM full_dataset_all_blur_3
WHERE noised == 1 AND discretization == 'float' and model == 'wiener_nonblind_noise' and blur_type == 'motion_blur'
ORDER BY psnr DESC, ssim DESC
LIMIT 20;



-- Select the BEST restoration examples for DWDN with MOTION BLUR
SELECT blur_type, kernel, image, psnr, ssim
FROM full_dataset_all_blur_3
WHERE noised == 1 AND discretization == 'float' and model == 'dwdn' and blur_type == 'motion_blur'
ORDER BY psnr DESC, ssim DESC
LIMIT 20;