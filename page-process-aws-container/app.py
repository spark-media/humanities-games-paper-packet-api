from __future__ import print_function
import traceback
import boto3
import copy
import cv2
import numpy as np
import json
import urllib.parse
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity

MAX_MATCHES = 500
GOOD_MATCH_PERCENT = 0.15
s3 = boto3.client('s3')
#dynamodb = boto3.client('dynamodb')
# This function is triggered by a page being inserted into a bucket.
# The key of the image is used to pull a template.
def perspectiveFixRequest(event, context):

    #print("Received event: " + json.dumps(event, indent=2))
    # What we hope to return.
    response = {
        "success": False,
        "message": "h.g v2.0!"
    }
    # Tasks
    # Triggered by S3 upload event to raw_images
    try:
        for record in event['Records']:
            # The S3 bucket
            bucket = record['s3']['bucket']['name']
            
            # Folder format is raw_assets/game/page/uuid. (jpg or jpeg or png)
            key = urllib.parse.unquote_plus( record['s3']['object']['key'], encoding='utf-8' )
            
            # Fetch Image
            page = s3.get_object(Bucket=bucket, Key=key)

            # page['Body'] is the image. Always .jpg
            # obj to cv2
            nparr = np.frombuffer( page['Body'].read(), np.uint8 )
            image = cv2.imdecode( nparr, cv2.IMREAD_COLOR )
            
            # Split the url into parts so we know where the model lives.
            # 1 is the minigame, 2 is the page, 3 is the uuid
            key_parts = key.split("/")
            minigame = key_parts[1]
            model_id = key_parts[2]
            image_id = key_parts[3].split(".")[0]

            # Fetch the page model for alignment.
            model_base = minigame + '/' + model_id
            model_bucket = 'humanities.games.page.models'
            page_model = s3.get_object( Bucket=model_bucket, Key=model_base+'.jpg' )
            model_nparr = np.frombuffer( page_model['Body'].read(), np.uint8 )
            model_image = cv2.imdecode( model_nparr, cv2.IMREAD_COLOR )

            # Fetch the page data for transforms.
            page_data_file = s3.get_object( Bucket=model_bucket, Key=model_base+'.json' )
            page_data_string = page_data_file["Body"].read().decode('utf-8')
            page_data = json.loads( page_data_string )
            

            # Align the page
            aligned_page = align_images( image, model_image )
            aligned_page = white_balance( aligned_page );

            # Locate and save the drawings.
            found_art = find_art( aligned_page, page_data, image_id, minigame, model_id, record );
            
            table = boto3.resource('dynamodb').Table('HG_API_PageStatus')
            # Update Dynamo Status to Complete based on UUID TODO.
            table.update_item(
                Key={
                    'SubmissionID': image_id
                },
                UpdateExpression="set ProcessStep = :s, FoundArt = :a",
                ExpressionAttributeValues={
                    ':s': "complete",
                    ':a': found_art
                },
                ReturnValues="UPDATED_NEW"
                )
            print("saved to dynamo")

    except Exception:
        # Caught but with error
        errorTrace = traceback.format_exc()
        print(errorTrace)
    finally:
        result = {
            "statusCode": 200,
            "body": json.dumps(response)
        }
        return result

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def align_images(im1, im2):

  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
  
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_MATCHES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
  descriptors1 = np.array(descriptors1)
  descriptors2 = np.array(descriptors2)

  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  # Match descriptors.
  matches = matcher.match(descriptors1,descriptors2, None)
  # Sort them in the order of their distance.
  matches = sorted(matches, key = lambda x:x.distance)

  # Sort matches by score
  #matches.sort(key=lambda x: x.distance, reverse=False)

  # Remove poor matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]

  # Draw top matches (local use)
  #imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  #cv2.imwrite("hg-matches.jpg", imMatches)
  
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
  
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))
  
  return im1Reg

def parse_transforms( image, block ):
    #transform
    if( block['rotate'] == "left" ) :
        image = cv2.rotate( image, cv2.ROTATE_90_COUNTERCLOCKWISE )

    if( block['rotate'] == "right" ) :
        image = cv2.rotate( image, cv2.ROTATE_90_CLOCKWISE )

    #resize
    new_size = ( block['scaleX'], block['scaleY'] )
    resized = cv2.resize( image, new_size, interpolation = cv2.INTER_AREA )
    return resized

def scale_contour( cnt, scale ):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)
    return cnt_scaled

def remove_paper( image ):
    # add padding
    image = cv2.copyMakeBorder(image,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])
    # converting image into grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      
    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
      
    # using a findContours() function
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      
    i = 0
    largestContourArea = 0
    largestContour = 0
    for cnt in contours:
        # findcontour function detects whole image as shape
        if i == 0:
            i = 1
            continue

        contourArea = cv2.contourArea(cnt)
        if( contourArea > largestContourArea):
            largestContour = cnt
            largestContourArea = contourArea

    if not isinstance( largestContour, int ):
        mask = np.zeros_like(image) # Create mask where white is what we want, black otherwise
        cv2.drawContours(image=mask, contours=[largestContour], contourIdx=-1, color=(255,255,255), thickness=-1) # Draw filled contour in mask
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        transparent = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        transparent[:,:,0:3] = image
        transparent[:, :, 3] = mask

        return transparent
    return image


def find_art( image, page_data, image_id, minigame, model_id, record ):

    #Save what is completed to an array
    completed = []
    completed_dict = {}
    #make a gray version
    gray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
    # setting threshold of gray image
    _, threshold = cv2.threshold( gray, 127, 255, cv2.THRESH_BINARY )
      
    # using a findContours() function
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
      
    i = 0
    total = len( page_data['blocks'] )
    # Find the biggest shapes.
    sorted_contours = sorted( contours, key=lambda x: cv2.contourArea(x), reverse=True )

    # Stored for later
    bucket = record['s3']['bucket']['name']
    image_key_base = 'processed/'+ minigame + '/' + model_id + '/' + image_id + '/'
    # was 40000 3-21-22
    threshold_area = 5000     #threshold area
    # list for storing names of shapes
    for contour in sorted_contours:
      
        # here we are ignoring first counter because 
        # findcontour function detects whole image as shape
        if i == 0:
            i = 1
            continue

        # quit if we are done.
        if len(completed_dict) >= total:
            break
        # cv2.approxPloyDP() function to approximate the shape
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)
          
        area = cv2.contourArea(contour)
        # Catch the biggest squares
        print( str(len(completed_dict)) + " of " + str(total) )
        if ( len(approx) == 4 ) and (area > threshold_area) and ( len(completed_dict) < total ):
            # use a point from page_data to determine which block 
            file_name = 'not-found'
            #see which contour 
            for block in page_data['blocks']:
                #is our known point inside the found box?
                result = cv2.pointPolygonTest(contour, ( block['centerX'], block['centerY'] ), False)
                if result > 0 :
                    newBlock = copy.deepcopy( block )
                    file_name = newBlock['file_name']
                    print( file_name + " found." )
                    # page_data['blocks'][ index ]['found'] = True;
                    # Scale the contour to try to remove the black border.
                    # .933 was the original working param here.
                    cnt = scale_contour( contour, .933 )
                    x,y,w,h = cv2.boundingRect( cnt )
                    cropped = image[y:y+h, x:x+w]
                    if block['make_transparent'] == True:
                        # format for js
                        newBlock['make_transparent'] = 'true'
                        # remove the paper
                        cropped = remove_paper( cropped )
                        final = parse_transforms( cropped, newBlock )
                        not_empty, rect = check_not_empty( final, file_name )
                        if not_empty == True :
                            # save to s3. The rect is for use with collision detection.
                            image_string = cv2.imencode( '.png', final, [int(cv2.IMWRITE_PNG_COMPRESSION),9] )[1].tostring()
                            s3.put_object(Bucket=bucket, Key=image_key_base + file_name + '.png', Body=image_string)
                            newBlock[ 'bounds' ] = rect
                            completed_dict[ file_name ] = newBlock;
                            #completed.append( json.dumps( newBlock, separators=(',', ':') ) )
                    else:
                        # format for js
                        newBlock['make_transparent'] = 'false'
                        # No transparency, let us save on file size
                        final = parse_transforms( cropped, newBlock )
                        not_empty, rect = check_not_empty( final, file_name )
                        if not_empty == True :
                            # save to s3. We disregard the rect for jpg
                            image_string = cv2.imencode( '.jpg', final, [cv2.IMWRITE_JPEG_QUALITY, 90] )[1].tostring()
                            s3.put_object(Bucket=bucket, Key=image_key_base + file_name + '.jpg', Body=image_string)
                            completed_dict[ file_name ] = newBlock;
                            #completed.append( json.dumps( newBlock, separators=(',', ':') ) )
    # Theoretically it takes the smallest match.
    for key in completed_dict:
        completed.append( json.dumps( completed_dict[ key ], separators=(',', ':') ) )
    return completed

def check_not_empty( image, file_name ):
    #make a gray version
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
      
    # using a findContours() function
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Small pencil marks etc TODO may require scale checking if not empty
    threshold_area = 1000  
    real_image = False

    if len(contours) <= 1 :
        # 1 is the whole image, so this is functionally blank.
        print( file_name + " is blank." )
        return False, {}
    else:
        # At least two, so something is in the box.
        largest_mark = sorted( contours, key=lambda x: cv2.contourArea(x), reverse=True )[ 1 ]
        area = cv2.contourArea( largest_mark )

        # Check if it is a big enough mark
        if area < threshold_area:
            print( file_name + "- drawing too small." )
            return False, {}
        # big enough, return bounding rect as array for collision detection 
        x,y,w,h = cv2.boundingRect( largest_mark )
        rect = { "x" : x, "y" : y, "width" : w, "height": h }
        return True, rect

def brightness_and_contrast( image, gray ):

    clip_hist_percent = 1
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    # Creates whiter background
    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return auto_result


  

