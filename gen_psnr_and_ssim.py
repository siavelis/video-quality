import os
import numpy
import imageio
from psnr import psnr
from ssim import ssim_exact
from glob import glob

from skimage.measure import compare_ssim

# ref_fileA = 'C:\\Users\\prs\\repos\\sese_thesis\\testing\\compare_nets\\output\\31364_lr_0.0100_l1_msk_file\\input.png'
# refA = imageio.imread(ref_fileA).astype(numpy.float32)
# print(numpy.shape(refA))

# ref_filePRS = 'C:\\Users\\prs\\repos\\sese_thesis\\testing\\compare_nets\\output\\31364_lr_0.0100_l1_msk_file\\proposed.png'
# refPRS = imageio.imread(ref_filePRS).astype(numpy.float32)

# ref_fileMoodoki = 'C:\\Users\\prs\\repos\\sese_thesis\\testing\\compare_nets\\output\\31364_lr_0.0100_l1_msk_file\\raymond.png'
# refMoodoki = imageio.imread(ref_fileMoodoki).astype(numpy.float32)

# _psnrPRS = psnr(refA, refPRS)
# _psnrMoodoki = psnr(refA, refMoodoki)
# print([_psnrPRS, _psnrMoodoki])

# print([ssim_exact(refA, refPRS), ssim_exact(refA, refMoodoki)])

inDir = 'C:\\Users\\ricca\\OneDrive\\Education\\SESE\\thesis\\experiments\\celebA\\input'
imgExt = 'png'
outDir = 'C:\\Users\\ricca\\OneDrive\\Education\\SESE\\thesis\\experiments\\celebA\\output_center_mask'
prefix = 'results\\celeba\\center_'
#outDir = 'C:\\Users\\ricca\\OneDrive\\Education\\SESE\\thesis\\experiments\\celebA\\output_random_mask'
#prefix = 'results\\celeba\\random_'
imgSize = [64, 64]

# compare_ssim(img, img_noise,data_range=img_noise.max() - img_noise.min())
psnrs = []
ssims = []
imgfilenames = glob(inDir + '/*.' + imgExt)
len_ = len(imgfilenames)
i = 0
for imgfilename in imgfilenames:
    lastIndex = imgfilename.rfind("\\")
    fileName = imgfilename[lastIndex + 1:]
    fileName = fileName[fileName.index('_') + 1:]

    #  ref_img = scipy.misc.imresize(imageio.imread(
    #     imgfilename), size=imgSize, interp='cubic')

    ref_img = imgfilename
    ref_img = imageio.imread(ref_img)

    ref_prsFile = outDir + '\\prs\\inpaint_' + fileName
    ref_prsFile = imageio.imread(ref_prsFile)
    ref_moodokiFile = outDir + '\\moodoki\\inpaint_' + fileName
    ref_moodokiFile = imageio.imread(ref_moodokiFile)

    _psnrPRS = psnr(ref_img, ref_prsFile)
    _psnrMoodoki = psnr(ref_img, ref_moodokiFile)

    tuple_ = [fileName, _psnrPRS, _psnrMoodoki]
    psnrs.append(tuple_)
    # print(tuple_)

    ref_img = ref_img / 255.0
    ref_prsFile = ref_prsFile / 255.0

    ref_moodokiFile = ref_moodokiFile / 255.0

    # _ssimPRS = ssim_exact(ref_img, ref_prsFile)
    _ssimPRS = compare_ssim(ref_img, ref_prsFile, data_range=ref_prsFile.max() - ref_prsFile.min(), multichannel=True)

    # _ssimMoodoki = ssim_exact(ref_img, ref_moodokiFile)
    _ssimMoodoki = compare_ssim(ref_img, ref_moodokiFile, data_range=ref_moodokiFile.max() - ref_moodokiFile.min(),
                                multichannel=True)

    tuple_ = [fileName, _ssimPRS, _ssimMoodoki]
    ssims.append(tuple_)
    # print(tuple_)
    # break
    i = i + 1
    print('%d/%d \r' % (i, len_), end='')

psnrPrs = []
psnrMoodoki = []
with open(prefix + 'prsn_results.csv', mode='wt', encoding='utf-8') as f:
    f.write('Frame ID, Proposed PSRN, Raymond PSNR\n')
    for i in range(len(psnrs)):
        value = psnrs[i]
        psnrPrs.append(value[1])
        psnrMoodoki.append(value[2])
        f.write('%s,%f,%f' % (value[0], value[1], value[2]) + '\n')

ssimPrs = []
ssimMoodoki = []
with open(prefix + 'ssims_results.csv', mode='wt', encoding='utf-8') as f:
    f.write('Frame ID, Proposed SSIM, Raymond SSIM\n')
    for i in range(len(ssims)):
        value = ssims[i]
        ssimPrs.append(value[1])
        ssimMoodoki.append(value[2])
        f.write('%s,%f,%f' % (value[0], value[1], value[2]) + '\n')

psnrPrsMean = numpy.mean(psnrPrs)
psnrMoodokiMean = numpy.mean(psnrMoodoki)
print('PSNR mean \nPRS: %f\nMoodoki: %f' % (psnrPrsMean, psnrMoodokiMean))

with open(prefix + 'psnr_results_mean.csv', mode='wt', encoding='utf-8') as f:
    f.write('# of Frames, Proposed PSRN (mean), Raymond PSRN (mean)\n')
    f.write('%d,%f,%f' % (i, psnrPrsMean, psnrMoodokiMean) + '\n')

ssimPrsMean = numpy.mean(ssimPrs)
ssimMoodokiMean = numpy.mean(ssimMoodoki)
print('SSIM mean \nPRS: %f\nMoodoki: %f' % (ssimPrsMean, ssimMoodokiMean))

with open(prefix + 'ssims_results_mean.csv', mode='wt', encoding='utf-8') as f:
    f.write('# of Frames, Proposed SSIM (mean), Raymond SSIM (mean)\n')
    f.write('%d,%f,%f' % (i, ssimPrsMean, ssimMoodokiMean) + '\n')
