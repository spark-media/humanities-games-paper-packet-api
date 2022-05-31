'use strict';
const uuid = require('uuid')
const { DynamoDBClient, GetItemCommand, PutItemCommand, UpdateItemCommand  } = require("@aws-sdk/client-dynamodb");
const { S3Client, PutObjectCommand, GetObjectCommand } = require('@aws-sdk/client-s3')
const { getSignedUrl } = require('@aws-sdk/s3-request-presigner')
/* Region setup */
const s3Client = new S3Client({region: 'us-east-1'});
const dynamoClient = new DynamoDBClient({ region: "us-east-1" });


module.exports.submitPageProcessRequest = async event => {

  /* What we expect to return and defaults */
  var response = {
    type : 'default',
    page : 'default',
    upload_url : '',
    success : false,
    submission_id : null,
    error : null
  }

  /* So our response is not broken if there is an error. */
  try {

    /* Pull path parameters to send it to the right spot. */
    if( typeof event['pathParameters'].minigameType !== 'undefined' ) {
      response.type = event['pathParameters'].minigameType;
    }
    if( typeof event['pathParameters'].minigamePage !== 'undefined' ) {
      response.page = event['pathParameters'].minigamePage;
    }

    /* A unique ID for this request */
    response.submission_id = uuid.v4();

    /* Create a signed URL the page will be uploaded to. */
    /* Only jpg is supported at this time. */
    var params = {
      Bucket: 'humanities.games.uploaded.pages',
      Key: 'raw_assets/' + response.type + '/' + response.page + '/' + response.submission_id + '.jpg',
      ContentType: 'image/jpeg'
    };

    /* Try just this part. */
    try {
      var command = new PutObjectCommand( params );
      response.upload_url = await getSignedUrl( s3Client, command, { expiresIn: 300 } )

      /* Create an entry in DynamoDB to track progress. */
      var now = Math.floor(Date.now() / 1000)
      /* 1 day */
      var expires = ( now + 86400 ).toString();
      var dbParams = {
        TableName: "HG_API_PageStatus",
        Item: {
          SubmissionID: { S: response.submission_id },
          ProcessStep: { S: "uploading" },
          Expires: { N:  expires },
          Minigame: { S: response.type },
          ModelID: { S: response.page }
        }
      }
      const data = await dynamoClient.send( new PutItemCommand( dbParams ) );
      console.log("Tracking record created.");

    } catch(e) {
      console.log('upload could not be generated', e )
      response.error = e;
    }

  } catch( mainError) {
    response.error = mainError;
    /* Something has gone awry */
    console.log( mainError )
  }

  return {
    statusCode: 200,
    body: JSON.stringify(
      response,
      null,
      2
    ),
  };
};

/* Status */
module.exports.checkPageProcessRequest = async event => {
  /* What we expect to return and defaults */
  var response = {
    type : 'default',
    page : 'default',
    status : 'processing',
    assets : [],
    success : false,
    error : null,
    submission_id : 0
  }

  /* So our response is not broken if there is an error. */
  try {

    /* Pull path parameters to send it to the right spot. */
    if( typeof event['pathParameters'].minigameType !== 'undefined' ) {
      response.type = event['pathParameters'].minigameType;
    }
    if( typeof event['pathParameters'].minigamePage !== 'undefined' ) {
      response.page = event['pathParameters'].minigamePage;
    }
    response.submission_id = event['pathParameters'].requestID

    /* Look up DynamoDB status. If done, generate URLs to download */
    var dbParams = {
      TableName: "HG_API_PageStatus", 
      Key: {
        SubmissionID: { S: response.submission_id }
      }
    }
    var data = await dynamoClient.send( new GetItemCommand( dbParams ) );
    if( typeof  data.Item !== 'undefined' ) {
      console.log( 'db', JSON.stringify( data.Item ))
      /* Entry found */
      if(  typeof data.Item.ProcessStep !== 'undefined' ) {
        response.status = data.Item.ProcessStep.S

        /* Check FoundArt if ProcessStep is complete */
        if(  response.status == 'complete' ) {

          /* Complete */
          if( typeof data.Item.FoundArt !== 'undefined' 
            && typeof data.Item.FoundArt.L !== 'undefined' 
            && data.Item.FoundArt.L.length > 0 ) {

            /* Generate a signed URL for each download. */
            for( var i = 0; i < data.Item.FoundArt.L.length; i++ ) {

              /* The game key */
              var block = JSON.parse( data.Item.FoundArt.L[ i ].S )
              var extension = 'jpg'
              /* 
              Boolean is a string to be valid json 
              (from Python's True/False capitalization) 
              */
              if( typeof block.make_transparent !== 'undefined'
                && block.make_transparent == 'true' ) {
                extension = 'png'
              }
              var params = {
                Bucket: 'humanities.games.uploaded.pages',
                Key: 'processed/' + response.type + '/' + response.page + '/' + response.submission_id + '/' + block.file_name + '.' + extension
              };

              /* Try just this part. */
              try {
                var command = new GetObjectCommand( params );
                /* Liberal 1 hour expiry. */
                var asset_url = await getSignedUrl( s3Client, command, { expiresIn: 3600 } )
                block.url = asset_url
                response.assets.push( block )

              } catch(e) {
                console.log('download could not be generated', e )
                response.error = e;
              }
            }
          } else {
            response.error = 'Art not found.'
          }
        }
      }
    } else {
      response.error = 'Record not found.'
    }

  } catch( mainError) {
    response.error = mainError;
    /* Something has gone awry */
    console.log( mainError )
  }

  return {
    statusCode: 200,
    body: JSON.stringify(
      response,
      null,
      2
    ),
  };
};
