import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def get_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

def get_roi(landmarks):
    knee, hip, ankle = [],[],[]
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    if(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z<landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z):
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    else:
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    return knee, hip, ankle, shoulder

cap = cv2.VideoCapture('KneeBendVideo.mp4')
#initializing variables
fps = cap.get(cv2.CAP_PROP_FPS)
kneeBent, frames, reps, fluctuation, feedback = 0,0,0,0,0
previousAngle = 180
cooldown = fps*2
width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
output = cv2.VideoWriter('Result.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (int(width), int(height)))

with mp_pose.Pose(min_detection_confidence=0.9, model_complexity=1, min_tracking_confidence=0.9) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        if(ret==True):
        
            #frame is processed through mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

            try:
                knee, hip, ankle, shoulder = get_roi(results.pose_landmarks.landmark)
                angle = get_angle(hip, knee, ankle)

                if(angle<140):
                    if(kneeBent!=1):
                        kneeBent = 1
                        frames += 1
                        fluctuation = 0
                    else:
                        frames+=1
                        if(frames==fps*8):
                            reps+=1
                            feedback = 1
                        fluctuation = 0
                else:
                    if(angle-previousAngle<20):
                        if(kneeBent==1):
                            if(frames<fps*8):
                                feedback = 2
                        kneeBent = 0
                        frames = 0
                        fluctuation = 0
                    elif(frames>=fps*8):
                        kneeBent = 0
                        frames = 0
                        fluctuation = 0
                    elif(angle-previousAngle>15):
                        if(fluctuation<30):
                            fluctuation +=1
                        else:
                            kneeBent = 0
                            frames = 0
                            fluctuation = 0
                    else:
                        if(kneeBent==1):
                            if(frames<fps*8):
                                feedback = 2
                        kneeBent = 0
                        fluctuation=0
                        frames = 0

                if(fluctuation==0):
                    previousAngle=angle

                cv2.putText(image, str(round(angle,4)), tuple(np.multiply(shoulder, [640, 480]).astype(int)), 
                     cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str(round(frames/fps,2)), (int(width-100), 70), 
                    cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2, cv2.LINE_AA)
                if(feedback==1):
                    cv2.putText(image, "Rep completed", (int(width/2)-100, 70), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2, cv2.LINE_AA)
                elif(feedback==2):
                    cv2.putText(image, "Keep you knee bent", (int(width/2)-100, 70), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2, cv2.LINE_AA)

                cv2.putText(image, "Reps: "+str(reps), (30, 70), 
                    cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2, cv2.LINE_AA)


            except:
                pass

            output.write(image)
            cv2.imshow('Knee Bend Rep Counter', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    output.release()
    cv2.destroyAllWindows()
