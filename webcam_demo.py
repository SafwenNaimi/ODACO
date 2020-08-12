import tensorflow as tf
import cv2
import time
import argparse
import math
import posenet
import numpy as np
import imutils 
import time

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=432)
parser.add_argument('--cam_height', type=int, default=368)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

def angle_between_points( p0, p1, p2 ):
  a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
  b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
  c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
  form=math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180 /math.pi
  return(form)


  

def main():
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        start = time.time()
        frame_count = 0
        init_time = time.time()
        test_timeout = init_time+10
        final_timeout = init_time+10
        counter_timeout_text = init_time+1
        counter_timeout = init_time+1
        counter = 10
        while True:
            #keyboard = np.zeros((1000, 1000, 3), np.uint8)
            pnts = []
            pntss = []
            pntsss = []
            pntssss = []
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)
            #print(keypoint_coords)

            keypoint_coords *= output_scale
            color=(0,0,255)
            # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)
            for pi in range(len(pose_scores)):
                
               


                if pose_scores[pi] == 0.:
                    break
                #print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                    if (posenet.PART_NAMES[ki]=='leftShoulder') or (posenet.PART_NAMES[ki]=='leftElbow') or (posenet.PART_NAMES[ki]=='leftWrist'):
                        pnts.append((int((c[0]) * 432 + 0.5), int((c[1])* 368 + 0.5)))
                    if (posenet.PART_NAMES[ki]=='rightShoulder') or (posenet.PART_NAMES[ki]=='rightElbow') or (posenet.PART_NAMES[ki]=='rightWrist'):
                        pntss.append((int((c[0]) * 432 + 0.5), int((c[1])* 368 + 0.5)))
                    if (posenet.PART_NAMES[ki]=='leftHip') or (posenet.PART_NAMES[ki]=='leftKnee') or (posenet.PART_NAMES[ki]=='leftAnkle'):
                        pntsss.append((int((c[0]) * 432 + 0.5), int((c[1])* 368 + 0.5)))
                    if (posenet.PART_NAMES[ki]=='rightHip') or (posenet.PART_NAMES[ki]=='rightKnee') or (posenet.PART_NAMES[ki]=='rightAnkle'):
                        pntssss.append((int((c[0]) * 432 + 0.5), int((c[1])* 368 + 0.5)))
                   
                    #print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))
                #print(pnts)
                angle = angle_between_points(pnts[0], pnts[1], pnts[2])
                anglee = angle_between_points(pntss[0], pntss[1], pntss[2])
                angleee = angle_between_points(pntsss[0], pntsss[1], pntsss[2])
                angleeee = angle_between_points(pntssss[0], pntssss[1], pntssss[2])
                print("angle left elbow",angle)
                print("angle right elbow",anglee)
                print("angle left knee",angleee)
                print("angle right knee",angleeee)


                if (165<(round(angle)) < 180):
                    color=(0,255,0)
                    
                else:
                    color=(0,0,255)

                if (165<(round(anglee)) < 180):
                    colors=(0,255,0)
                else:
                    colors=(0,0,255)

                if (90<(round(angleee)) < 140):
                    colorss=(0,255,0)
                else:
                    colorss=(0,0,255)

                if (165<(round(angleeee)) < 180):
                    colorsss=(0,255,0)
                else:
                    colorsss=(0,0,255)

                cv2.putText(overlay_image,"left elbow ",(10,90),1,2,color,2)
                cv2.putText(overlay_image,"right elbow ",(10,120),1,2,colors,2)
                cv2.putText(overlay_image,"left knee ",(10,150),1,2,colorss,2)
                cv2.putText(overlay_image,"right knee ",(10,180),1,2,colorsss,2)
                
                if (time.time() > counter_timeout_text and time.time() < test_timeout):
                    cv2.putText(
                    overlay_image, "Counter: "+str(counter), (10,50), 1, 2, (255,255,0), 2)
                    print(counter)
                    counter_timeout_text+=0.03333
                if (time.time() > counter_timeout and time.time() < test_timeout):
                    counter-=1
                    counter_timeout+=1
                


                #cv2.putText(overlay_image,"left elbow "+str(round(angle)),(10,90),1,2,color,2)
                #cv2.putText(overlay_image,"right elbow "+str(round(anglee)),(10,120),1,2,color,2)
                #cv2.putText(overlay_image,"left knee "+str(round(angleee)),(10,150),1,2,color,2)
                #cv2.putText(overlay_image,"right knee "+str(round(angleeee)),(10,180),1,2,color,2)

            im = cv2.imread("saff.png")
            im = cv2.resize(im, (400, 360))
            #keyboard = cv2.resize(keyboard, (150, 662))
            key=np.hstack([overlay_image,im])
            key = imutils.resize(key, height=600)
            #cv2.putText(key,"left elbow ",(10,90),1,2,color,2)
            #cv2.putText(key,"right elbow ",(10,120),1,2,colors,2)
            #cv2.putText(key,"left knee ",(10,150),1,2,colorss,2)
            #cv2.putText(key,"right knee ",(10,180),1,2,colorsss,2)
            #cv2.imshow("Result",key)

            cv2.imshow('Output', key)
            
            frame_count += 1
            cv2.imwrite('saf.jpg',key)
            
            
            if (cv2.waitKey(1) & 0xFF == ord('q')) or (time.time() > final_timeout):
                break

        #image=cv2.imread('out.png')
        #cv2.imshow('outp',image)
        #cv2.waitKey(0)

        print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()
