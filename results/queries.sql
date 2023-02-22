-- SQLite
-- Select all data stored in the database
SELECT *
FROM full_test;

---------------------------------- DATASET EXPLORATION ----------------------------------

-- Select blur types
SELECT blur_type, COUNT(DISTINCT kernel)
FROM full_test
GROUP BY blur_type;


-- Select images
SELECT image_dataset, COUNT(DISTINCT image)
FROM full_test
GROUP BY image_dataset;


---------------------------------- MODEL BENCHMARKING ----------------------------------

-- Select mean metrics for different model types
SELECT discretization, blur_type, noised, model, AVG(psnr), AVG(ssim)
FROM full_test
GROUP BY discretization, blur_type, noised, model;


-- Select mean metrics for different model types for all blur types (not quite correct, because not all models were tested on gauss_blur or eye_blur)
SELECT noised, model, AVG(psnr), AVG(ssim)
FROM full_test
GROUP BY noised, model;


--motion blur (type of blur which was used to testing with all models)
SELECT discretization, noised, model, AVG(psnr), AVG(ssim)
FROM full_test
WHERE blur_type == 'motion_blur'
GROUP BY discretization, noised, model;


----------------------------- float -----------------------------

-- Select mean metrics for different model types and FLOAT dicretization 
-- (not quite correct, because not all models were tested on gauss_blur or eye_blur)
SELECT noised, model, AVG(psnr), AVG(ssim)
FROM full_test
WHERE discretization == 'float'
GROUP BY noised, model


-- Select mean metrics for different model types + FLOAT dicretization + MOTION_BLUR
SELECT noised, model, AVG(psnr), AVG(ssim)
FROM full_test
WHERE discretization == 'float' and blur_type == 'motion_blur'
GROUP BY noised, model


-- more noise (sigma 0.02)
SELECT noised, model, AVG(psnr), AVG(ssim)
FROM full_test_more_noise
WHERE discretization == 'float' and blur_type == 'motion_blur'
GROUP BY noised, model


----------------------------- srgb_8bit -----------------------------

-- Select mean metrics for different model types and UINT8 dicretization
-- (not quite correct, because not all models were tested on gauss_blur or eye_blur)
SELECT noised, model, AVG(psnr), AVG(ssim)
FROM full_test
WHERE discretization == 'srgb_8bit'
GROUP BY noised, model


-- Select mean metrics for different model types | FLOAT dicretization | MOTION_BLUR
SELECT noised, model, AVG(psnr), AVG(ssim)
FROM full_test
WHERE discretization == 'srgb_8bit' and blur_type == 'motion_blur'
GROUP BY noised, model


-- more noise (sigma 0.02)
SELECT noised, model, AVG(psnr), AVG(ssim)
FROM full_test_more_noise
WHERE discretization == 'srgb_8bit' and blur_type == 'motion_blur'
GROUP BY noised, model


----------------------------- linrgb_16bit -----------------------------

-- Select mean metrics for different model types | FLOAT dicretization | MOTION_BLUR
SELECT noised, model, AVG(psnr), AVG(ssim)
FROM full_test
WHERE discretization == 'linrgb_16bit' and blur_type == 'motion_blur'
GROUP BY noised, model


---------------------------------- WIENER ----------------------------------

----------------------------- worst restoration -----------------------------

-- Select the WORST restoration examples for WIENER
SELECT blur_type, kernel, image, ssim, psnr
FROM full_test
WHERE noised == 0 AND discretization == 'float' and model == 'wiener_nonblind_noise'
ORDER BY psnr, ssim
LIMIT 20;


SELECT blur_type, kernel, image, ssim, psnr
FROM full_test
WHERE noised == 0 AND discretization == 'float' and model == 'wiener_blind_noise'
ORDER BY psnr, ssim
LIMIT 20;

-- Select the WORST restoration examples for WIENER
SELECT kernel, image, ssim, psnr
FROM full_test
WHERE noised == 0 AND discretization == 'float' and model == 'wiener_nonblind_noise' and blur_type == 'motion_blur'
ORDER BY psnr, ssim
LIMIT 20;


----------------------------- best restoration -----------------------------

-- float
SELECT blur_type, kernel, image, psnr, ssim
FROM full_test
WHERE noised == 0 AND discretization == 'float' and model == 'wiener_nonblind_noise'
ORDER BY psnr DESC, ssim DESC
LIMIT 20;

-- srgb_8bit
SELECT blur_type, kernel, image, psnr, ssim
FROM full_test
WHERE noised == 0 AND discretization == 'srgb_8bit' and model == 'wiener_nonblind_noise'
ORDER BY psnr DESC, ssim DESC
LIMIT 20;

-- linrgb_16bit
SELECT blur_type, kernel, image, psnr, ssim
FROM full_test
WHERE noised == 0 AND discretization == 'linrgb_16bit' and model == 'wiener_nonblind_noise'
ORDER BY psnr DESC, ssim DESC
LIMIT 20;


-- Select the BEST restoration examples for WIENER with GAUSS BLUR
SELECT blur_type, kernel, image, psnr, ssim
FROM full_test
WHERE noised == 0 AND discretization == 'float' and model == 'wiener_nonblind_noise' and blur_type == 'gauss_blur'
ORDER BY psnr DESC, ssim DESC
LIMIT 20;