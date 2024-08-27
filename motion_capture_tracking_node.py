#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import geometry_msgs
from motion_capture_tracking_interfaces.msg import NamedPoseArray, NamedPose
from tf2_ros import TransformBroadcaster
import numpy as np
import struct
import time
from scipy import linalg
import cv2 as cv
import copy
import os
import threading
from pseyepy import Camera
from scipy.spatial.transform import Rotation as Rotation
from scipy.signal import butter, lfilter
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.duration import Duration
import math





class Cameras:
    def __init__(self):
        self.cameras = Camera(fps=90, resolution=Camera.RES_SMALL, gain=25, exposure=100)
        self.num_cameras = 4

        self.camera_params_prev = [
                {
                    "intrinsic_matrix": [[272.02344764,   0.0,         153.19869958],
             [  0.0,         272.06718365, 123.01836245],
             [  0.0,           0.0,           1.0        ]],

                                        "distortion_coef":[-0.11003612,  0.20722031,  0.0007618,  -0.00254586, -0.11575396],
                    "rotation": 2
                },
                {
                    "intrinsic_matrix": [[278.80691686,   0.0,         165.63093555],
             [  0.0,         278.53210894, 115.7635302 ],
             [  0.0,           0.0,           1.0        ]],


                                        "distortion_coef":[-0.1399076,   0.27003803,  0.00160763,  0.00343932, -0.18160045],
                    "rotation": 0
                },
                {
                    "intrinsic_matrix": [[281.80198909,   0.0,         161.65886593],
             [  0.0,         281.75087464, 122.25908036],
             [  0.0,           0.0,           1.0        ]],
                
                                         "distortion_coef":[-0.12494047,  0.31513948, -0.00124578,  0.00150465, -0.28071058],
                    "rotation": 2
                },
                {
                    "intrinsic_matrix": [[256.27048958,   0.0,         150.57131994],
             [  0.0,         256.14052794, 115.55580242],
             [  0.0,           0.0,           1.0        ]],
                       
                                         "distortion_coef":[-0.12259003,  0.16819705,  0.00318028, -0.00249815, -0.070743  ],
                    "rotation": 0
                }
            ]

        self.camera_params = []
        self.camera_poses = [{"R":[[1,0,0],[0,1,0],[0,0,1]],"t":[0,0,0]},{"R":[[-0.12237129460736368,-0.32923113668747056,0.9362863477011688],[0.24259206964916008,0.9048388775703596,0.34987954124319554],[-0.962379527005565,0.26995085529164,-0.030857442017399705]],"t":[-3.309908865383017,-1.0108903093929158,2.614082650466829]},{"R":[[-0.9991002587758974,0.03959288569337644,0.015201194571672582],[0.040381509051474355,0.7785451383499434,0.6262881136333283],[0.01296173756331748,0.6263384635755163,-0.7794434696661466]],"t":[0.028688502626513214,-1.7787388303094998,6.040751357038269]},{"R":[[-0.10858850243695889,0.31522757648460065,-0.9427831734615036],[-0.1857502652986269,0.9252535296191082,0.3307608575222763],[0.9765784024234757,0.211039050772664,-0.041918289194391556]],"t":[3.3881646162941745,-0.9621128111657591,2.3545102169485834]}]
        #[{"R":[[1,0,0],[0,1,0],[0,0,1]],"t":[0,0,0]},{"R":[[-0.11949841987010063,-0.3253777664681683,0.9380028980428233],[0.2785711480038362,0.8958330385401436,0.34623876524698727],[-0.9529523824139242,0.3026755294835043,-0.01640977463301635]],"t":[-3.6163587498013494,-1.2440082569866755,2.7820286125282423]},{"R":[[-0.9973624787139882,0.06435690171070084,0.03356002466757516],[0.07111322386872042,0.7739193158066903,0.6292787951395639],[0.01452568223823974,0.6300056204697725,-0.7764547138964449]],"t":[0.00003739188629271501,-2.1003790108188163,6.463393526745379]},{"R":[[-0.09955137743464448,0.33237595002646214,-0.9378783242483384],[-0.13917348059678608,0.9286441725498742,0.3438760577413542],[0.9852513717571982,0.16476112597686954,-0.04618988869214952]],"t":[3.663303566001363,-1.2550450415692036,2.446761453285444]}]
        #self.camera_poses = [{"R":[[1,0,0],[0,1,0],[0,0,1]],"t":[0,0,0]},{"R":[[-0.46781741897842993,-0.16245115220511208,0.8687672217847494],[0.14633988157315975,0.9551577068349484,0.25740705921741336],[-0.8716257806888884,0.24755479842054137,-0.4230663307537617]],"t":[-0.5609466380726404,-0.10633774578575474,0.991155143879089]},{"R":[[-0.9964583574544098,0.06024022221713112,-0.05866734599834915],[0.03020133222237515,0.9075316551215055,0.4188963767859712],[0.07847688443900584,0.41564096354865127,-0.9061368373645867]],"t":[0.1273086152374389,-0.3016757444357096,1.333536817552464]},{"R":[[0.3410572589628937,0.16882608202511315,-0.9247581846822269],[-0.19563057688583077,0.9749496063147711,0.10583922965512449],[0.9194610505595303,0.14481373959235672,0.3655411841766156]],"t":[0.6803637517589269,-0.0772930248336733,0.42335422228503605]}]

                                        
        for i in range(self.num_cameras):
            self.camera_params.append({
                "intrinsic_matrix": np.array(self.camera_params_prev[i]["intrinsic_matrix"]),
                "distortion_coef": np.array(self.camera_params_prev[i]["distortion_coef"]),
                "rotation": self.camera_params_prev[i]["rotation"]
            })

        self.num_objects = 1
        
        self.to_world_coords_matrix = [[0.9986197375825863,0.05201306684487569,-0.007297985192989492,0.15281404170364876],[-0.05201306684487567,0.9600324463507569,-0.2750133502780763,2.551855347269922],[0.00729798519298949,-0.2750133502780763,-0.9614127087681704,3.8915595393513986],[0,0,0,1]]
    

    def read(self):
        start = time.time()
        frames, _ = self.cameras.read()

        
        for i in range(0, self.num_cameras):
            frames[i] = np.rot90(frames[i], k=self.camera_params[i]["rotation"])
            frames[i] = self.make_square(frames[i])
            frames[i] = cv.undistort(frames[i], self.get_camera_params(i)["intrinsic_matrix"], self.get_camera_params(i)["distortion_coef"])
            frames[i] = cv.GaussianBlur(frames[i],(9,9),0)
            kernel = np.array([[-2,-1,-1,-1,-2],
                               [-1,1,3,1,-1],
                               [-1,3,4,3,-1],
                               [-1,1,3,1,-1],
                               [-2,-1,-1,-1,-2]])
            frames[i] = cv.filter2D(frames[i], -1, kernel)
            frames[i] = cv.cvtColor(frames[i], cv.COLOR_RGB2BGR)




        image_points = []
        for i in range(0, self.num_cameras):
            frames[i], single_camera_image_points = self._find_dot(frames[i])
            #print(single_camera_image_points)
            image_points.append(single_camera_image_points)
            
        if (any(np.all(point[0] != [None,None]) for point in image_points)):

            errors, object_points, frames = self.find_point_correspondance_and_object_points(image_points, self.camera_poses, frames)

                    # convert to world coordinates
            for i, object_point in enumerate(object_points):
                new_object_point = np.array([[-1,0,0],[0,-1,0],[0,0,1]]) @ object_point
                new_object_point = np.concatenate((new_object_point, [1]))
                new_object_point = np.array(self.to_world_coords_matrix) @ new_object_point
                new_object_point = new_object_point[:3] / new_object_point[3]
                new_object_point[1], new_object_point[2] = new_object_point[2], new_object_point[1]
                object_points[i] = new_object_point

                    #print(object_points)
                    #if len(object_points) == 1:
                    #    return {'pos': object_points[0], 'quaternion': [0.0, 0.0, 0.0, 1.0]}
                    #else:
                    #    return None
                    # Left Marker: highest y
                    # Right Marker: lowest y
                    # Front Marker: highest x (or remaining)
                    #print(time.time() - start)

                   
            if len(object_points) == 3:
                for element in object_points:
                    element[0], element[1] = -element[1], element[0]

                if object_points[0][1] > object_points[1][1] and object_points[0][1] > object_points[2][1]:
                    L = np.array(object_points[0])
                    if object_points[1][1] < object_points[2][1]:
                        R = np.array(object_points[1])
                        F = np.array(object_points[2])
                    else: 
                        R = np.array(object_points[2])
                        F = np.array(object_points[1])

                elif object_points[1][1] > object_points[0][1] and object_points[1][1] > object_points[2][1]:
                    L = np.array(object_points[1])
                    if object_points[0][1] < object_points[2][1]:
                        R = np.array(object_points[0])
                        F = np.array(object_points[2])
                    else: 
                        R = np.array(object_points[2])
                        F = np.array(object_points[0])
                elif object_points[2][1] > object_points[0][1] and object_points[2][1] > object_points[1][1]:
                    L = np.array(object_points[2])
                    if object_points[0][1] < object_points[1][1]:
                        R = np.array(object_points[0])
                        F = np.array(object_points[1])
                    else: 
                        R = np.array(object_points[1])
                        F = np.array(object_points[0])

                        #print(f'Front: {F}, Right: {R}, Left: {L}')

                def ensure_positive_direction(vector, reference):
                    if np.dot(vector, reference) < 0:
                        vector = -vector
                    return vector




                y_axis = L - R # from right to left
                y_axis = y_axis / np.linalg.norm(y_axis)

                v = F - L # random in plane
                z_axis = np.cross(y_axis, v)
                z_axis = z_axis / np.linalg.norm(z_axis) 
                z_axis = ensure_positive_direction(z_axis, np.array([0,0,1]))

                x_axis = np.cross(z_axis, y_axis)
                x_axis = x_axis / np.linalg.norm(x_axis)
                x_axis = ensure_positive_direction(x_axis, np.array([1,0,0]))



                    #y_axis = ensure_negative_direction(y_axis, np.array([0,1,0]))


                        

                    #x_axis = ensure_positive_direction(x_axis, np.array([1,0,0]))
                        

                Rot = np.vstack([x_axis, y_axis, z_axis]).T


                    #rotate = np.array([[0.0, -1.0, 0.0],
                    #    [1.0, 0.0, 0.0],
                    #    [0.0 ,0.0, 1.0]])

                    #Rot = np.dot(Rot, rotate)

                roll = math.atan2(Rot[2,1], Rot[2,2])
                pitch = math.atan2(-Rot[2,0], math.sqrt(Rot[2,1]**2 + Rot[2,2]**2))
                yaw = math.atan2(Rot[1,0], Rot[0,0])

                    #print(yaw)


                object_position = F# + R + L / 3#(R + L + F) / 3
                        #print(f'Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}')
                        #print(object_position, ';', roll, pitch, yaw)

                ro = Rotation.from_matrix(Rot)
                quaternion = ro.as_quat()

                    #print('Front: ', F)
                    #print('Obj: ', F + R + L / 3)

                        



                return {'pos': object_position.tolist(), 'quaternion': quaternion}


                        #working

                        """if object_points[0][0] > object_points[1][0] and object_points[0][0] > object_points[2][0]:
                            L = np.array(object_points[0])
                            if object_points[1][0] < object_points[2][0]:
                                R = np.array(object_points[1])
                                F = np.array(object_points[2])
                            else: 
                                R = np.array(object_points[2])
                                F = np.array(object_points[1])

                        elif object_points[1][0] > object_points[0][0] and object_points[1][0] > object_points[2][0]:
                            L = np.array(object_points[1])
                            if object_points[0][0] < object_points[2][0]:
                                R = np.array(object_points[0])
                                F = np.array(object_points[2])
                            else: 
                                R = np.array(object_points[2])
                                F = np.array(object_points[0])

                        elif object_points[2][0] > object_points[0][0] and object_points[2][0] > object_points[1][0]:
                            L = np.array(object_points[2])
                            if object_points[0][0] < object_points[1][0]:
                                R = np.array(object_points[0])
                                F = np.array(object_points[1])
                            else: 
                                R = np.array(object_points[1])
                                F = np.array(object_points[0])

                        #print(f'Front: {F}, Right: {R}, Left: {L}')

                        def ensure_positive_direction(vector, reference):
                            if np.dot(vector, reference) < 0:
                                vector = -vector
                            return vector

                        def ensure_negative_direction(vector, reference):
                            if np.dot(vector, reference) > 0:
                                vector = -vector
                            return vector


                        x_axis = L - R # from right to left
                        x_axis = x_axis / np.linalg.norm(x_axis)

                        v = F - L # random in plane
                        z_axis = np.cross(x_axis, v)
                        z_axis = z_axis / np.linalg.norm(z_axis) 
                        z_axis = ensure_positive_direction(z_axis, np.array([0,0,1]))

                        y_axis = np.cross(z_axis, x_axis)
                        y_axis = y_axis / np.linalg.norm(y_axis)



                        #y_axis = ensure_negative_direction(y_axis, np.array([0,1,0]))


                        

                        #x_axis = ensure_positive_direction(x_axis, np.array([1,0,0]))
                        

                        Rot = np.vstack([x_axis, y_axis, z_axis]).T


                        rotate = np.array([[0.0, 1.0, 0.0],
                            [-1.0, 0.0, 0.0],
                            [0.0 ,0.0, 1.0]])

                        Rot = np.dot(Rot, rotate)

                        roll = math.atan2(Rot[2,1], Rot[2,2])
                        pitch = math.atan2(-Rot[2,0], math.sqrt(Rot[2,1]**2 + Rot[2,2]**2))
                        yaw = math.atan2(Rot[1,0], Rot[0,0])

                        print(yaw)


                        object_position = F#(R + L + F) / 3
                        #print(f'Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}')
                        #print(object_position, ';', roll, pitch, yaw)

                        ro = Rotation.from_matrix(Rot)
                        quaternion = ro.as_quat()

                        



                        return {'pos': object_position.tolist(), 'quaternion': quaternion}"""

                    """objects = []
                    filtered_objects = []
                    if self.is_locating_objects:
                        objects, quaternion = self.locate_objects(object_points, errors)
                        

                        filtered_objects = self.kalman_filter.predict_location(objects)

                        if len(objects) != 0:
                            #print('Normal Objects: ', objects[0]['pos'].tolist())
                            #print('Filtered Objects: ', filtered_objects[0]['pos'].tolist())
                            return {'pos': objects[0]['pos'].tolist(), 'quaternion': [0.0, 0.0, 0.0, 1.0]}

                        if len(objects) == 0:
                            if self.prev_object != None:
                                filtered_objects = self.kalman_filter.predict_location(self.prev_object)
                            else:
                                pass

                        else:
                            filtered_objects = self.kalman_filter.predict_location(objects)
                            self.prev_object = filtered_objects


                        #print('objects:', objects)
                        #print('object_points:', object_points)
                        #print('filtered_objects:', filtered_objects)
                        #if len(objects) != 0:
                        #    print(objects[0]['pos'])
                        #    return {'pos': objects[0]['pos'].tolist(), 'quaternion': [0.0, 0.0, 0.0, 1.0]}

                        
                        
                        if len(filtered_objects) != 0:


                            for filtered_object in filtered_objects:
                                
                                pass

                                #if self.drone_armed[filtered_object['droneIndex']]:
                                #    filtered_object["heading"] = round(filtered_object["heading"], 4)
                        
                            
                        for filtered_object in filtered_objects:
                            filtered_object["vel"] = filtered_object["vel"].tolist()
                            filtered_object["pos"] = filtered_object["pos"].tolist()
                    


                        
                    
                    #if len(object_points) != 0:

                        #print(object_points)
                        #return {'pos': object_points[0], 'quaternion': [0.0, 0.0, 0.0, 1.0]}"""

        return None#{'pos': [0.0, 0.0, 0.0], 'quaternion': [0.0, 0.0, 0.0, 1.0]}

    def get_frames(self):
        frames = self._camera_read()

        return np.hstack(frames)

    def _find_dot(self, img):
        grey = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        grey = cv.threshold(grey, 255*0.2, 255, cv.THRESH_BINARY)[1]
        contours,_ = cv.findContours(grey, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        img = cv.drawContours(img, contours, -1, (0,255,0), 1)

        image_points = []
        for contour in contours:
            moments = cv.moments(contour)
            if moments["m00"] != 0:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
                cv.putText(img, f'({center_x}, {center_y})', (center_x,center_y - 15), cv.FONT_HERSHEY_SIMPLEX, 0.3, (100,255,100), 1)
                cv.circle(img, (center_x,center_y), 1, (100,255,100), -1)
                image_points.append([center_x, center_y])

        if len(image_points) == 0:
            image_points = [[None, None]]
        

        return img, image_points

    
    def get_camera_params(self, camera_num):
        return {
            "intrinsic_matrix": np.array(self.camera_params[camera_num]["intrinsic_matrix"]),
            "distortion_coef": np.array(self.camera_params[camera_num]["distortion_coef"]),
            "rotation": self.camera_params[camera_num]["rotation"]
        }
    


    def triangulate_point(self, image_points, camera_poses):
        image_points = np.array(image_points)

        none_indicies = np.where(np.all(image_points == None, axis=1))[0]
        image_points = np.delete(image_points, none_indicies, axis=0)
        camera_poses = np.delete(camera_poses, none_indicies, axis=0)

        if len(image_points) <= 1:
            return [None, None, None]

        Ps = [] # projection matricies

        for i, camera_pose in enumerate(camera_poses):
            RT = np.c_[camera_pose["R"], camera_pose["t"]]
            P = self.camera_params[i]["intrinsic_matrix"] @ RT
            Ps.append(P)

        def DLT(Ps, image_points): # Direct Linear Transform
            A = []

            for P, image_point in zip(Ps, image_points):
                A.append(image_point[1]*P[2,:] - P[1,:])
                A.append(P[0,:] - image_point[0]*P[2,:])
                
            A = np.array(A).reshape((len(Ps)*2,4))
            B = A.transpose() @ A
            U, s, Vh = linalg.svd(B, full_matrices = False)
            object_point = Vh[3,0:3]/Vh[3,3]

            return object_point

        object_point = DLT(Ps, image_points)

        return object_point


    def triangulate_points(self, image_points, camera_poses):
        object_points = []
        for image_points_i in image_points:
            object_point = self.triangulate_point(image_points_i, camera_poses)
            object_points.append(object_point)
        
        return np.array(object_points)


    def find_point_correspondance_and_object_points(self, image_points, camera_poses, frames):

        for image_points_i in image_points:
            try:
                image_points_i.remove([None, None])
            except:
                pass

        if len(image_points[0]) == 3 and len(image_points[1]) == 3 and len(image_points[2]) == 3 and len(image_points[3]) == 3:
        #if True:
            # [object_points, possible image_point groups, image_point from camera]
            correspondances = [[[i]] for i in image_points[0]]

            Ps = [] # projection matricies
            for i, camera_pose in enumerate(camera_poses):
                RT = np.c_[camera_pose["R"], camera_pose["t"]]
                P = self.camera_params[i]["intrinsic_matrix"] @ RT
                Ps.append(P)

            root_image_points = [{"camera": 0, "point": point} for point in image_points[0]]

            for i in range(1, len(camera_poses)):
                epipolar_lines = []
                for root_image_point in root_image_points:
                    F = cv.sfm.fundamentalFromProjections(Ps[root_image_point["camera"]], Ps[i])
                    line = cv.computeCorrespondEpilines(np.array([root_image_point["point"]], dtype=np.float32), 1, F)
                    epipolar_lines.append(line[0,0].tolist())
                    frames[i] = self.drawlines(frames[i], line[0])

                not_closest_match_image_points = np.array(image_points[i])
                points = np.array(image_points[i])

                for j, [a, b, c] in enumerate(epipolar_lines):
                    distances_to_line = np.array([])
                    if len(points) != 0:
                        distances_to_line = np.abs(a*points[:,0] + b*points[:,1] + c) / np.sqrt(a**2 + b**2)

                    possible_matches = points[distances_to_line < 15].copy() # 20 # 10 otherwise longer computation

                    # sort possible matches from smallest to largest
                    distances_to_line = distances_to_line[distances_to_line < 15] # changed from < 5, otherwise multiple points found which leads to bad result
                    possible_matches_sorter = distances_to_line.argsort()
                    possible_matches = possible_matches[possible_matches_sorter]
            
                    if len(possible_matches) == 0:
                        for possible_group in correspondances[j]:
                            possible_group.append([None, None])
                    else:
                        not_closest_match_image_points = [row for row in not_closest_match_image_points.tolist() if row != possible_matches.tolist()[0]]
                        not_closest_match_image_points = np.array(not_closest_match_image_points)
                        
                        new_correspondances_j = []
                        for possible_match in possible_matches:
                            temp = copy.deepcopy(correspondances[j])
                            for possible_group in temp:
                                possible_group.append(possible_match.tolist())
                            new_correspondances_j += temp
                        correspondances[j] = new_correspondances_j


            object_points = []
            errors = []
            
            for image_points in correspondances:
                object_points_i = self.triangulate_points(image_points, camera_poses)

                if np.all(object_points_i == None):
                    continue

                errors_i = self.calculate_reprojection_errors(image_points, object_points_i, camera_poses)

                object_points.append(object_points_i[np.argmin(errors_i)])
                errors.append(np.min(errors_i))

            #print([image_points, object_points])



            return np.array(errors), np.array(object_points), frames

        return np.array([]), np.array([]), frames



    def numpy_fillna(self, data):
        data = np.array(data, dtype=object)
        # Get lengths of each row of data
        lens = np.array([len(i) for i in data])

        # Mask of valid places in each row
        mask = np.arange(lens.max()) < lens[:,None]

        # Setup output array and put elements from data into masked positions
        out = np.full((mask.shape[0], mask.shape[1], 2), [None, None])
        out[mask] = np.concatenate(data)
        return out
            

    def drawlines(self, img1,lines):
        r,c,_ = img1.shape
        for r in lines:
            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        return img1


    def make_square(self, img):
        x, y, _ = img.shape
        size = max(x, y)
        new_img = np.zeros((size, size, 3), dtype=np.uint8)
        ax,ay = (size - img.shape[1])//2,(size - img.shape[0])//2
        new_img[ay:img.shape[0]+ay,ax:ax+img.shape[1]] = img

        # Pad the new_img array with edge pixel values
        # Apply feathering effect
        feather_pixels = 8
        for i in range(feather_pixels):
            alpha = (i + 1) / feather_pixels
            new_img[ay - i - 1, :] = img[0, :] * (1 - alpha)  # Top edge
            new_img[ay + img.shape[0] + i, :] = img[-1, :] * (1 - alpha)  # Bottom edge


        return new_img



class MotionCaptureTrackingNode(Node):
    def __init__(self):
        super().__init__('motion_capture_tracking_node')
        self.declare_parameter('hostname', 'localhost')
        self.declare_parameter('logfilepath', '')

        self.hostname = self.get_parameter('hostname').get_parameter_value().string_value
        self.logfilepath = self.get_parameter('logfilepath').get_parameter_value().string_value

        self.mocap = Cameras()

        self.pub_point_cloud = self.create_publisher(PointCloud2, 'pointCloud', 1)

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,  # The queue size is set to 50
            deadline=Duration(seconds=0, nanoseconds=int(1e9 / 50.0))  # 10 Hz deadline
        )

        self.pub_poses = self.create_publisher(NamedPoseArray, 'cf1_pose', qos_profile)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.timer = self.create_timer(0.01, self.update)

        self.dt = time.time()


        self.prev = {'crazyflie': {'position': np.array([0.0, 0.0, 0.0]), 'orientation': np.array([0.0, 0.0, 0.0, 1.0])}}

    def update(self):


        obs = self.mocap.read()

        if obs == None:
            rigid_bodies = self.prev
        else:
                

            rigid_bodies = {
                'cf1': {
                    'position': np.array(obs["pos"]),
                    'orientation': np.array(obs["quaternion"])
                }
            }

            self.prev = rigid_bodies

        self.publish_poses(rigid_bodies)



    def publish_poses(self, rigid_bodies):
        msg = NamedPoseArray()
        msg.header.frame_id = 'world'
        msg.header.stamp = self.get_clock().now().to_msg()
        
        transforms = []

        for name, data in rigid_bodies.items():
            pose = NamedPose()
            pose.name = name
            pose.pose.position.x = data['position'][0]
            pose.pose.position.y = data['position'][1]
            pose.pose.position.z = data['position'][2]
            pose.pose.orientation.x = data['orientation'][0]
            pose.pose.orientation.y = data['orientation'][1]
            pose.pose.orientation.z = data['orientation'][2]
            pose.pose.orientation.w = data['orientation'][3]
            msg.poses.append(pose)

            transform = geometry_msgs.msg.TransformStamped()
            transform.header.stamp = msg.header.stamp
            transform.header.frame_id = 'world'
            transform.child_frame_id = name
            transform.transform.translation.x = data['position'][0]
            transform.transform.translation.y = data['position'][1]
            transform.transform.translation.z = data['position'][2]
            transform.transform.rotation.x = data['orientation'][0]
            transform.transform.rotation.y = data['orientation'][1]
            transform.transform.rotation.z = data['orientation'][2]
            transform.transform.rotation.w = data['orientation'][3]
            transforms.append(transform)

        self.pub_poses.publish(msg)
        self.tf_broadcaster.sendTransform(transforms)

def main(args=None):
    rclpy.init(args=args)
    node = MotionCaptureTrackingNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()