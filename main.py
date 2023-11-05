# importing required libraries
import cv2
import mediapipe as mp
import numpy as np

# Accessing camera
cap = cv2.VideoCapture(0)
draw = mp.solutions.drawing_utils

# access the height and width of the screen or frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_shape = (frame_height,frame_width,3)

prevxy = None
mask = np.zeros(frames_shape,dtype='uint8')
colour = (123,34,90)
thickness = 4

hands = mp.solutions.hands
hand_landmark = hands.Hands(max_num_hands=1)

tools = cv2.imread("tool.png")
tools = tools.astype('uint8')

midCol = frame_width // 2
max_row = 50
min_col = midCol-125
max_col = midCol+125

curr_tool = "draw"

start_point = None

def check_exit(key):
    if key == 27:  # ESC key
        return True
    return False

# Check if distance between 2 points is less than 60 pixels
def get_is_clicked(point1, point2):
    (x1, y1) = point1
    (x2, y2) = point2
    
    dis = (x1-x2)**2 + (y1-y2)**2
    dis = np.sqrt(dis)
    if dis<60:
        return True
    else:
        return False

# Return tool based on column location
def get_Tool(point, prev_tool):
    (x, y) = point
    
    if x>min_col and x<max_col and y<max_row:
        if x < min_col:
            return
        elif x < 50 + min_col:
            curr_tool = "line"
        elif x<100 + min_col:
            curr_tool = "rectangle"
        elif x < 150 + min_col:
            curr_tool ="draw"
        elif x<200 + min_col:
            curr_tool = "circle"
        else:
            curr_tool = "erase"
        return curr_tool
    else:
        return prev_tool

# Camera reads window till interruption
while True:
  # captures video frame by frame
  _, frame = cap.read()
  frame = cv2.flip(frame, 1) 
  
  rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  op = hand_landmark.process(rgb)
  
  if op.multi_hand_landmarks:
    for all_landmarks in op.multi_hand_landmarks:
      draw.draw_landmarks(frame,all_landmarks,hands.HAND_CONNECTIONS)
      
      # index finger location
      x = int(all_landmarks.landmark[8].x*frames_shape[1])
      y = int(all_landmarks.landmark[8].y*frames_shape[0])
            
      # Middle finger location
      middle_x = all_landmarks.landmark[12].x * frames_shape[1]
      middle_y = all_landmarks.landmark[12].y * frames_shape[0]
      middle_x, middle_y = int(middle_x), int(middle_y)
      
      is_clicked = get_is_clicked((x, y), (middle_x, middle_y))
      curr_tool = get_Tool((x, y), curr_tool)

      # Select tool and draw for that
      if curr_tool == 'draw':
          # Connect previous and current index finger locations
          if is_clicked and prevxy!=None:
              cv2.line(mask, prevxy, (x, y), colour, thickness)
      
      elif curr_tool == 'rectangle':
          if is_clicked and start_point == None:
              start_point = (x, y)
          
          elif is_clicked:
              cv2.rectangle(frame, start_point, (x, y), colour, thickness)
          
          elif is_clicked == False and start_point:
              cv2.rectangle(mask, start_point, (x, y), colour, thickness)
              start_point=None
              
      elif curr_tool=='circle':
                if is_clicked and start_point==None:
                    start_point = (x, y)
                
                if start_point:
                    rad = int(((start_point[0]-x)**2 + (start_point[1]-y)**2)**0.5)
                if is_clicked:
                    cv2.circle(frame, start_point, rad, colour, thickness)
                
                if is_clicked==False and start_point:
                    cv2.circle(mask, start_point, rad, colour, thickness)
                
                    start_point=None
            
      elif curr_tool == "erase":
          cv2.circle(frame, (x, y), 30, (0,0,0), -1) # -1 means fill
          if is_clicked:
              cv2.circle(mask, (x, y), 30, 0, -1)
              
      prevxy = (x,y)
      
  frame = np.where(mask,mask,frame)
  
  frame[0:max_row, min_col:max_col] = tools
  
  # displays img in window
  cv2.imshow('Live', frame)
  # Enter esc char to terminate
  key = cv2.waitKey(1)
  if check_exit(key):
      break
# release the resources
cap.release()
cv2.destroyAllWindows()
