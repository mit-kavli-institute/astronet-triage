import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
import tess_stars2px
from tess_stars2px import tess_stars2px_function_entry

sys.path.insert(1, 'transit-diffImage')
from transitDiffImage import tessDiffImage, transitCentroids
from transitDiffImage import tessprfmodel as tprf



def process_astro_id(astro_id):
    ## Get TIC information for a given Astro ID
    star = {}
    star['id'] = table['TIC ID'][astro_id]
    star['raDegrees'] = table['RA'][astro_id]
    star['decDegrees'] = table['Dec'][astro_id]

    planet0 = {}
    planet0['planetID'] = f"astroid{astro_id}"
    planet0['period'] = table['Per'][astro_id]
    planet0['epoch'] = table['Epoc'][astro_id]
    planet0['durationHours'] = table['Dur'][astro_id] * 24

    ## Create directory to save images and redirect stdout and stderr to outputs.log
    dirname = f"tic-images/tic{star['id']}"
    os.makedirs(dirname, exist_ok=True)
   
    # try:
    if os.path.exists(f'{dirname}/centroid_distance_astroid{astro_id}.txt'):
        print(f"Already done this Astro ID ({astro_id}, (TIC {star['id']}), skipping...")
        return

    if np.isnan(table['RA'][astro_id]):
        print(f"Skipping Astro ID ({astro_id}) (TIC {star['id']} with NaN entries")
        return
    # except Exception as e:
    #     print(e)
    #     return
   

    with open(f'{dirname}/outputs.log','wt') as outputs_f:
        #sys.stdout = outputs_f
        #sys.stderr = outputs_f
        print(f"Astro ID: {astro_id}\n ---------------------------------")

        ## Use TESSpoint to get which sectors TIC is observed in
        outID, outEclipLong, outEclipLat, outSec, outCam, outCcd, \
            outColPix, outRowPix, scinfo = tess_stars2px_function_entry(star['id'], 
                                                                        star['raDegrees'], 
                                                                        star['decDegrees'])
        print(outSec, outCam, outCcd)
        
        for sectorIndex in range(len(outSec)):
            print(f"-------------------\nSector: {outSec[sectorIndex]}\n")

            # try:

            ## Use TESScut to download FFI cutout, and calculate difference image
            star['sector'] = outSec[sectorIndex]
            star['cam'] = outCam[sectorIndex]
            star['ccd'] = outCcd[sectorIndex]
            star['planetData'] = [planet0]
            star['qualityFiles'] = None 
            star['qualityFlags'] = None
            print(star)

            tdi = tessDiffImage.tessDiffImage(star, outputDir=os.path.dirname(dirname))
            tdi.make_ffi_difference_image(thisPlanet=0)

            ## Load Image Data
            fname = f"{dirname}/imageData_{planet0['planetID']}_sector{star['sector']}.pickle"
            if os.path.exists(fname):
                with open(fname, 'rb') as f:
                    imageData = pickle.load(f)
                diffImageData = imageData[0]
                catalogData = imageData[1]
            else:
                print(f"Error: {fname} does not exist")
                continue

            # Save Diff Images
            np.save(f"{dirname}/diffImage_{planet0['planetID']}_sector{star['sector']}.npy", diffImageData)

            # # Load Diff Images
            # imageData = np.load(f"{dirname}/diffImage_{planet0['planetID']}_sector{star['sector']}.npy", allow_pickle=True)
            # imageData = imageData[()]['diffImage']


            ## Calculate Centroids

            # Create TESS PRF object
            if ('diffImage' not in diffImageData) or (np.any(np.isnan(diffImageData['diffImage']))):
                print("Error: Difference Image not available or has NaNs")
                continue
            prf = tprf.SimpleTessPRF(shape=diffImageData["diffImage"].shape,
                                    sector = outSec[sectorIndex],
                                    camera = outCam[sectorIndex],
                                    ccd = outCcd[sectorIndex],
                                    column=catalogData["extent"][0],
                                    row=catalogData["extent"][2],
                                    # prfFileLocation = "../../tessPrfFiles/"
            )

            # Compute the centroid
            fitVector, prfFitQuality, fluxCentroid, closeDiffImage, closeExtent, opt_method = transitCentroids.tess_PRF_centroid(prf, 
                                                        catalogData["extent"], 
                                                        diffImageData["diffImage"], 
                                                        catalogData)
            
            print("PRF fit quality = " + str(prfFitQuality))


            # # Compute the centroid distance in pixels 1
            # centroidRa, centroidDec, scinfo = tess_stars2px.tess_stars2px_reverse_function_entry(tdi.sectorList[0], outCam[sectorIndex], outCcd[sectorIndex], fitVector[0], fitVector[1])
            # outID, centroidEclipLong, centroidEclipLat, centroidSec, centroidCam, centroidCcd, centroidColPix, centroidRowPix, scinfo = tess_stars2px.tess_stars2px_function_entry(0, centroidRa, centroidDec, aberrate=True, trySector=tdi.sectorList[0])
            # dCol = centroidColPix[0] - fitVector[0]
            # dRow = centroidRowPix[0] - fitVector[1]
            # d2 = dCol*dCol + dRow*dRow
            # centroid_distance_pix1 = str(np.sqrt(d2))
            # print("centroid_distance_pix1", centroid_distance_pix1)

            # # Compute the centroid distance in pixels 2
            # centroid_distance_pix2 = tessDiffImage.pix_distance([centroidRa, centroidDec], tdi.sectorList[0], outCam[sectorIndex], outCcd[sectorIndex], fitVector[0], fitVector[1])
            # print("centroid_distance_pix2", centroid_distance_pix2)

            # Compute the centroid in RA and Dec
            raDec = tessDiffImage.pix_to_ra_dec(tdi.sectorList[0], outCam[sectorIndex], outCcd[sectorIndex], fitVector[0], fitVector[1])

            dRa = raDec[0] - catalogData['correctedRa'][0]
            dDec = raDec[1] - catalogData['correctedDec'][0]
            centroid_distance_arcsec = str(3600*np.sqrt((dRa*np.cos(catalogData['correctedDec'][0]*np.pi/180))**2 + dDec**2))
            print("distance = " + centroid_distance_arcsec + " arcsec")


            # Plot difference image
            fig, ax = plt.subplots(2,2,figsize=(10,10))
            tdi.draw_pix_catalog(diffImageData['diffImage'], catalogData, catalogData["extent"], ax=ax[0,0], fs=14, ss=60, filterStars=True, dMagThreshold=4, annotate=True)
            tdi.draw_pix_catalog(diffImageData['diffImage'], catalogData, catalogData["extentClose"], ax=ax[0,1], fs=14, ss=60, filterStars=True, dMagThreshold=4, annotate=True, close=True)
            tdi.draw_pix_catalog(diffImageData['meanOutTransit'], catalogData, catalogData["extent"], ax=ax[1,0], fs=14, ss=60, filterStars=True, dMagThreshold=4, annotate=True)
            tdi.draw_pix_catalog(diffImageData['meanOutTransit'], catalogData, catalogData["extentClose"], ax=ax[1,1], fs=14, ss=60, filterStars=True, dMagThreshold=4, annotate=True, close=True)
            ax[0,0].set_title('Difference Image')
            ax[0,1].set_title('Difference Image (Close-up)')
            ax[1,0].set_title('Direct Image')
            ax[1,1].set_title('Direct Image (Close-up)')

            fig.suptitle(f"Centroid Distance: {centroid_distance_arcsec} arcsec")
            plt.savefig(f"{dirname}/diffImage_{planet0['planetID']}_sector{star['sector']}.png")
            plt.close() 

            
            # Show the flux-weighted and PRF-fit centroids on the difference image, along with the position of the target star (the first star in the catalog data).
            plt.imshow(closeDiffImage, cmap='jet', origin='lower', extent=closeExtent)
            plt.plot(fluxCentroid[0], fluxCentroid[1], 'w+', label = "flux-weighted centroid", zorder=200)
            plt.plot(fitVector[0], fitVector[1], 'ws', label = "PRF-fit centroid", zorder=200)
            plt.axvline(catalogData["targetColPix"][0], c='y', label = "target star")
            plt.axhline(catalogData["targetRowPix"][0], c='y')
            plt.colorbar()
            plt.legend()
            plt.title(f"Centroid Distance: {centroid_distance_arcsec} arcsec")
            plt.savefig(f"{dirname}/centroid_diffImage_{planet0['planetID']}_sector{star['sector']}.png")
            plt.close()

            # Save centroid distance to file
            with open(f'{dirname}/centroid_distance_astroid{astro_id}.txt', 'a') as f_centroids:
                f_centroids.write(f"{astro_id},{star['id']},{star['sector']},{centroid_distance_arcsec},{prfFitQuality},{opt_method}\n")

            # except Exception as e:
            #     print("###########\n Error:", e, "\n############")

        

def create_centroid_distance_csv(astro_ids):
    with open('centroid_distance_astro_ids.csv', 'w') as f_combined:
        f_combined.write("Astro ID,TIC ID,Sector,Centroid Distance (arcsec),PRF Fit Quality,Optim Method\n")
        for astro_id in astro_ids:
            print(astro_id)
            dirname = f"tic-images/tic{table['TIC ID'][astro_id]}"
            filename = f'{dirname}/centroid_distance_astroid{astro_id}.txt'
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    data = f.readlines()#.strip().split(',')
                    f_combined.writelines(data)


def quick_flux_centroid(arr, extent, constrain=True):
    xpix = np.linspace(extent[0], extent[1]-1, arr.shape[1])
    ypix = np.linspace(extent[2], extent[3]-1, arr.shape[0])
    X, Y = np.meshgrid(xpix, ypix)
    normArr = arr.copy() - np.median(arr.ravel())
    sum_f = np.sum(normArr.ravel())
    sum_x = np.sum((X*normArr).ravel())
    sum_y = np.sum((Y*normArr).ravel())
    
    xc = sum_x/sum_f
    yc = sum_y/sum_f
    
    if constrain:
        # if the centroid is outside the extent then return the center of the image
        if (xc < extent[0]) | (xc > extent[1]):
            xc = np.mean(extent[0:2])

        if (yc < extent[2]) | (yc > extent[3]):
            yc = np.mean(extent[2:])

    return [xc, yc]


def main():
    # # Load Vetting csv file with list of TIC IDs
    # fpath = "/pdo/users/dmuth/mnt/tess/astronet/tces-vetting-v02-tois-triageJs-nocentroid-all.csv" # "/pdo/users/dmuth/mnt/tess/labels/vetting-v02.csv"
    # table = pd.read_csv(fpath, header=0, low_memory=False).set_index('Astro ID')

    min_id = int(sys.argv[1])
    max_id = int(sys.argv[2])
    
    # Get list of Astro IDs
    astro_ids =  table.index.values[min_id:max_id]  #
    print(astro_ids)
    print(len(astro_ids))

    # Process Astro IDs in parallel
    #num_processes = multiprocessing.cpu_count() - 3
    #with Pool(processes=num_processes) as pool:
    #  pool.map(process_astro_id, astro_ids)

    # # Serial processing
    for astro_id in astro_ids:
       print(f"Processing Astro ID: {astro_id} of {len(astro_ids)}")
       process_astro_id(astro_id)

    # Create centroid_distance_astro_ids.csv
    #create_centroid_distance_csv(astro_ids)

if __name__ == "__main__":
    # Load Vetting csv file with list of TIC IDs
    fpath = "/pdo/users/dmuth/mnt/tess/astronet/tces-vetting-v02-tois-triageJs-nocentroid-all.csv" # "/pdo/users/dmuth/mnt/tess/labels/vetting-v02.csv"
    table = pd.read_csv(fpath, header=0, low_memory=False).set_index('Astro ID')

    main()



    
