package com.example.aslreader;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Bundle;
import android.widget.TextView;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.ml.common.FirebaseMLException;
import com.google.firebase.ml.custom.FirebaseCustomLocalModel;
import com.google.firebase.ml.custom.FirebaseModelDataType;
import com.google.firebase.ml.custom.FirebaseModelInputOutputOptions;
import com.google.firebase.ml.custom.FirebaseModelInputs;
import com.google.firebase.ml.custom.FirebaseModelInterpreter;
import com.google.firebase.ml.custom.FirebaseModelInterpreterOptions;
import com.google.firebase.ml.custom.FirebaseModelOutputs;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.common.FirebaseVisionImageMetadata;
import com.otaliastudios.cameraview.CameraView;
import com.otaliastudios.cameraview.frame.Frame;
import com.otaliastudios.cameraview.frame.FrameProcessor;

public class MainActivity extends AppCompatActivity {

    FirebaseCustomLocalModel localModel = new FirebaseCustomLocalModel.Builder()
            .setAssetFilePath("converted_model.tflite")
            .build();

    FirebaseModelInterpreter interpreter;
    FirebaseModelInputOutputOptions inputOutputOptions;
    String[] out_label = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"};
    String sequenceString ="";
    String mem = "";
    Integer frameCount = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        CameraView cameraView = findViewById(R.id.cameraView);
        final TextView alphabet = findViewById(R.id.DetectedAlphabet);
        final TextView sequence = findViewById(R.id.constructedSequence);
        cameraView.setLifecycleOwner(this);

        cameraView.addFrameProcessor(
                new FrameProcessor() {
                    @Override
                    public void process(@NonNull Frame frame) {
                        interpreter = initializeInterpreter();
                        inputOutputOptions = initializeInputOutputOptions();
                        FirebaseVisionImage data = getImageFromFrame(frame);
                        Bitmap dataBitmap = data.getBitmap();
                        Bitmap bitmap = Bitmap.createScaledBitmap(dataBitmap,200,200,true);

                        int batchNum = 0;
                        float[][][][] input = new float[1][200][200][3];
                        for (int x = 0; x < 200; x++) {
                            for (int y = 0; y < 200; y++) {
                                int pixel = bitmap.getPixel(x, y);
                                input[batchNum][x][y][0] =  (Color.red(pixel)-127)/ 128.0f;
                                input[batchNum][x][y][1] = (Color.green(pixel)-127) / 128.0f;
                                input[batchNum][x][y][2] = (Color.blue(pixel)-127) / 128.0f;
                            }
                        }
                        FirebaseModelInputs inputs = null;
                        try {
                            inputs = new FirebaseModelInputs.Builder()
                                    .add(input)  // add() as many input arrays as your model requires
                                    .build();
                        } catch (FirebaseMLException e) {
                            e.printStackTrace();
                        }
                        interpreter.run(inputs, inputOutputOptions)
                                .addOnSuccessListener(
                                        new OnSuccessListener<FirebaseModelOutputs>() {
                                            @Override
                                            public void onSuccess(FirebaseModelOutputs result) {
                                                float[][] output = result.getOutput(0);
                                                float[] probabilities = output[0];
                                                float max = 0;
                                                int index = 0;
                                                for(int i =0;i<probabilities.length;i++)
                                                {
                                                    if(max<probabilities[i])
                                                    {
                                                        max = probabilities[i];
                                                        index = i;
                                                    }
                                                }
                                                String alpha = out_label[index];
                                                if(alpha.equals(mem))
                                                {
                                                    frameCount++;
                                                }
                                                else
                                                {
                                                    mem = alpha;
                                                }
                                                alphabet.setText(alpha);
                                                if(frameCount == 2)
                                                {
                                                    frameCount = 0;
                                                    if((sequenceString.length()>0) && alpha.equals("del"))
                                                    {
                                                        sequenceString.substring(0,sequenceString.length()-2);
                                                    }
                                                    else if( alpha.equals("space"))
                                                    {
                                                        sequenceString += " ";
                                                    }
                                                    else if(alpha.equals("nothing"))
                                                    {
                                                        sequenceString = sequenceString;
                                                    }
                                                    else
                                                    {
                                                        sequenceString+=alpha;
                                                    }
                                                    sequence.setText(sequenceString);
                                                }
                                            }
                                        })
                                .addOnFailureListener(
                                        new OnFailureListener() {
                                            @Override
                                            public void onFailure(@NonNull Exception e) {
                                                alphabet.setText("Failure");
                                            }
                                        });

                    }
                }
        );
    }

    protected FirebaseModelInterpreter initializeInterpreter(){
        FirebaseModelInterpreter interpreter = null;
        try {
            FirebaseModelInterpreterOptions options =
                    new FirebaseModelInterpreterOptions.Builder(localModel).build();
            interpreter = FirebaseModelInterpreter.getInstance(options);
        } catch (FirebaseMLException e) {
            e.printStackTrace();
        }
        return interpreter;

    }

    protected FirebaseModelInputOutputOptions initializeInputOutputOptions(){
        FirebaseModelInputOutputOptions inputOutputOptions = null ;
        try {
            inputOutputOptions =
                    new FirebaseModelInputOutputOptions.Builder()
                            .setInputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, 200, 200, 3})
                            .setOutputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, 29})
                            .build();
        } catch (FirebaseMLException e) {
            e.printStackTrace();
        }
        return inputOutputOptions;

    }

   protected FirebaseVisionImage getImageFromFrame(Frame frame)
   {
       byte[] data = frame.getData();


       FirebaseVisionImageMetadata imageMetaData = new FirebaseVisionImageMetadata.Builder()
               .setFormat(FirebaseVisionImageMetadata.IMAGE_FORMAT_NV21)
               .setRotation(FirebaseVisionImageMetadata.ROTATION_90)
               .setHeight(frame.getSize().getHeight())
               .setWidth(frame.getSize().getWidth())
               .build();

       FirebaseVisionImage image = FirebaseVisionImage.fromByteArray(data, imageMetaData);

       return image;



   }


}
