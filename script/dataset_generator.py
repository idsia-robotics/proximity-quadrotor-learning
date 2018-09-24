# ------ Import ------
import os
import sys
from multiprocessing import Pool

import pandas as pd
import rosbag
import tqdm as tqdm

from global_parameters import *
from direct_controller import Controller
from utils import *


class DatasetCreator:
    def __init__(self):
        """
            initializer for the class. creates an empty self.dataset
        """
        self.dataset = []

    def generate_data(self, data_vec):
        """
            append new data_vec to self.dataset
        Args:
            data_vec: vector of data
        """
        self.dataset += data_vec

    def save_dataset(self, flag_train, model_type, title="wrong.pickle"):
        """
            Save the dataset
        Args:
            model_type: dataset type
            flag_train: flag indicating the type of dataset to be saved
            title: name of the dateset file

        Returns:
            None if error in flag_train
        """
        if model_type == "v1":
            folder_n = "version1"
        elif model_type == "v2":
            folder_n = "version2"
        elif model_type == "v3":
            folder_n = "version3"
        elif model_type == "v4":
            folder_n = "green"
        else:
            print("ERROR in model type save_dataset")
            sys.exit(1)

        if flag_train == "train":
            train = pd.DataFrame(list(self.dataset))
            train.to_pickle("../dataset/" + folder_n + "/train.pickle")
        elif flag_train == "test":
            val = pd.DataFrame(list(self.dataset))
            val.to_pickle("../dataset/" + folder_n + "/test.pickle")
        else:
            print("ERROR in FLAG TRAIN")
            sys.exit(1)


def get_bag_data_pandas(bag, model_type):
    """
        Read a bag object and returns dictionary of dataframes
    Args:
        bag: bagfile object
        model_type: type of the model

    Returns:
        dictionary
    """
    if model_type == "v1":
        h_id = []
        h_v = []
        for topic, hat, t in bag.read_messages(topics=['/optitrack/head']):
            secs = t.secs
            nsecs = t.nsecs
            h_id.append(time_conversion_to_nano(secs, nsecs))
            pos_rot_dict = (lambda x, y: {'h_pos_x': x.x,
                                          'h_pos_y': x.y,
                                          'h_pos_z': x.z,
                                          'h_rot_w': y.w,
                                          'h_rot_x': y.x,
                                          'h_rot_y': y.y,
                                          'h_rot_z': y.z})(hat.pose.position, hat.pose.orientation)
            h_v.append(pos_rot_dict)
        head_df = pd.DataFrame(data=h_v, index=h_id, columns=h_v[0].keys())

        b_id = []
        b_v = []
        for topic, bebop, t in bag.read_messages(topics=['/optitrack/bebop']):
            secs = t.secs
            nsecs = t.nsecs
            b_id.append(time_conversion_to_nano(secs, nsecs))
            pos_rot_dict = (lambda x, y: {'b_pos_x': x.x,
                                          'b_pos_y': x.y,
                                          'b_pos_z': x.z,
                                          'b_rot_w': y.w,
                                          'b_rot_x': y.x,
                                          'b_rot_y': y.y,
                                          'b_rot_z': y.z})(bebop.pose.position, bebop.pose.orientation)
            b_v.append(pos_rot_dict)
        bebop_df = pd.DataFrame(data=b_v, index=b_id, columns=b_v[0].keys())
        c_id = []
        c_v = []
        for topic, image_frame, t in bag.read_messages(topics=['/bebop/image_raw/compressed']):
            secs = t.secs
            nsecs = t.nsecs
            c_id.append(time_conversion_to_nano(secs, nsecs))
            img = jpeg2np(image_frame.data, (image_width, image_height))
            camera_frame = (lambda x: {'vid': x})(img)
            c_v.append(camera_frame)
        camera_df = pd.DataFrame(data=c_v, index=c_id, columns=c_v[0].keys())

        odom_id = []
        odom_v = []

        for topic, msg, t in bag.read_messages(topics=['/bebop/odom']):  # OPTICAL FLOW
            vel = msg.twist
            results = [vel.twist.linear.x, vel.twist.linear.y]
            secs = t.secs
            nsecs = t.nsecs
            odom_id.append(time_conversion_to_nano(secs, nsecs))
            vel_dict = (lambda x: {'x_vel': x[0],
                                   'y_vel': x[1]})(results)
            odom_v.append(vel_dict)
        odom_df = pd.DataFrame(data=odom_v, index=odom_id, columns=odom_v[0].keys())
        bag.close()
        return {'head_df': head_df, 'bebop_df': bebop_df, 'camera_df': camera_df, 'odom_df': odom_df}
    elif model_type == "v2":
        bt_id = []
        bt_v = []
        for topic, beboptw, t in bag.read_messages(topics=['/bebop/mocap_odom']):
            secs = t.secs
            nsecs = t.nsecs
            bt_id.append(time_conversion_to_nano(secs, nsecs))
            twist_dict = (lambda x: {'t_x': x.x,
                                     't_y': x.y})(beboptw.twist.twist.linear)
            bt_v.append(twist_dict)
        bebop_twist_df = pd.DataFrame(data=bt_v, index=bt_id, columns=bt_v[0].keys())
        odom_id = []
        odom_v = []
        for topic, msg, t in bag.read_messages(topics=['/bebop/odom']):
            vel = msg.twist
            results = [vel.twist.linear.x, vel.twist.linear.y]
            secs = t.secs
            nsecs = t.nsecs
            odom_id.append(time_conversion_to_nano(secs, nsecs))
            vel_dict = (lambda x: {'x_vel': x[0],
                                   'y_vel': x[1]})(results)
            odom_v.append(vel_dict)
        odom_df = pd.DataFrame(data=odom_v, index=odom_id, columns=odom_v[0].keys())
        c_id = []
        c_v = []
        for topic, image_frame, t in bag.read_messages(topics=['/bebop/image_raw/compressed']):
            secs = t.secs
            nsecs = t.nsecs
            c_id.append(time_conversion_to_nano(secs, nsecs))
            img = jpeg2np(image_frame.data, (image_width, image_height))
            camera_frame = (lambda x: {'vid': x})(img)
            c_v.append(camera_frame)
        camera_df = pd.DataFrame(data=c_v, index=c_id, columns=c_v[0].keys())
        h_id = []
        h_v = []
        for topic, hat, t in bag.read_messages(topics=['/optitrack/head']):
            secs = t.secs
            nsecs = t.nsecs
            h_id.append(time_conversion_to_nano(secs, nsecs))
            pos_rot_dict = (lambda x, y: {'h_pos_x': x.x,
                                          'h_pos_y': x.y,
                                          'h_pos_z': x.z,
                                          'h_rot_w': y.w,
                                          'h_rot_x': y.x,
                                          'h_rot_y': y.y,
                                          'h_rot_z': y.z})(hat.pose.position, hat.pose.orientation)
            h_v.append(pos_rot_dict)
        head_df = pd.DataFrame(data=h_v, index=h_id, columns=h_v[0].keys())

        b_id = []
        b_v = []
        for topic, bebop, t in bag.read_messages(topics=['/optitrack/bebop']):
            secs = t.secs
            nsecs = t.nsecs
            b_id.append(time_conversion_to_nano(secs, nsecs))
            pos_rot_dict = (lambda x, y: {'b_pos_x': x.x,
                                          'b_pos_y': x.y,
                                          'b_pos_z': x.z,
                                          'b_rot_w': y.w,
                                          'b_rot_x': y.x,
                                          'b_rot_y': y.y,
                                          'b_rot_z': y.z})(bebop.pose.position, bebop.pose.orientation)
            b_v.append(pos_rot_dict)
        bebop_df = pd.DataFrame(data=b_v, index=b_id, columns=b_v[0].keys())
        bag.close()
        return {'head_df': head_df, 'bebop_df': bebop_df, 'odom_df': odom_df, 'camera_df': camera_df, 'bebop_twist_df': bebop_twist_df}
    elif model_type == "v3":
        bt_id = []
        bt_v = []
        for topic, beboptw, t in bag.read_messages(topics=['/bebop/mocap_odom']):
            secs = t.secs
            nsecs = t.nsecs
            bt_id.append(time_conversion_to_nano(secs, nsecs))
            twist_dict = (lambda x: {'t_x': x.x,
                                     't_y': x.y})(beboptw.twist.twist.linear)
            bt_v.append(twist_dict)
        bebop_twist_df = pd.DataFrame(data=bt_v, index=bt_id, columns=bt_v[0].keys())
        h_id = []
        h_v = []
        for topic, hat, t in bag.read_messages(topics=['/optitrack/head']):
            secs = t.secs
            nsecs = t.nsecs
            h_id.append(time_conversion_to_nano(secs, nsecs))
            pos_rot_dict = (lambda x, y: {'h_pos_x': x.x,
                                          'h_pos_y': x.y,
                                          'h_pos_z': x.z,
                                          'h_rot_w': y.w,
                                          'h_rot_x': y.x,
                                          'h_rot_y': y.y,
                                          'h_rot_z': y.z})(hat.pose.position, hat.pose.orientation)
            h_v.append(pos_rot_dict)
        head_df = pd.DataFrame(data=h_v, index=h_id, columns=h_v[0].keys())

        b_id = []
        b_v = []
        for topic, bebop, t in bag.read_messages(topics=['/optitrack/bebop']):
            secs = t.secs
            nsecs = t.nsecs
            b_id.append(time_conversion_to_nano(secs, nsecs))
            pos_rot_dict = (lambda x, y: {'b_pos_x': x.x,
                                          'b_pos_y': x.y,
                                          'b_pos_z': x.z,
                                          'b_rot_w': y.w,
                                          'b_rot_x': y.x,
                                          'b_rot_y': y.y,
                                          'b_rot_z': y.z})(bebop.pose.position, bebop.pose.orientation)
            b_v.append(pos_rot_dict)
        bebop_df = pd.DataFrame(data=b_v, index=b_id, columns=b_v[0].keys())

        odom_id = []
        odom_v = []
        for topic, msg, t in bag.read_messages(topics=['/bebop/odom']):
            vel = msg.twist
            results = [vel.twist.linear.x, vel.twist.linear.y]
            secs = t.secs
            nsecs = t.nsecs
            odom_id.append(time_conversion_to_nano(secs, nsecs))
            vel_dict = (lambda x: {'x_vel': x[0],
                                   'y_vel': x[1]})(results)
            odom_v.append(vel_dict)
        odom_df = pd.DataFrame(data=odom_v, index=odom_id, columns=odom_v[0].keys())
        c_id = []
        c_v = []
        for topic, image_frame, t in bag.read_messages(topics=['/bebop/image_raw/compressed']):
            secs = t.secs
            nsecs = t.nsecs
            c_id.append(time_conversion_to_nano(secs, nsecs))
            img = jpeg2np(image_frame.data, (image_width, image_height))
            camera_frame = (lambda x: {'vid': x})(img)
            c_v.append(camera_frame)
        camera_df = pd.DataFrame(data=c_v, index=c_id, columns=c_v[0].keys())
        bag.close()
        return {'head_df': head_df, 'bebop_df': bebop_df, 'odom_df': odom_df, 'camera_df': camera_df, 'bebop_twist_df': bebop_twist_df}


def processing(bag_df_dict, data_id, f, model_type):
    """
        Process data from dictionary bag_df_dict into a data vector
    Args:
        bag_df_dict: dictionary of Pandas dataframes
        data_id: id of the bag file processed
        f: bag file name, used as key for dictionary
        model_type: type of the model

    Returns:
        data vector
    """
    if model_type == "v1":
        camera_t = bag_df_dict["camera_df"].index.values
        bebop_t = bag_df_dict["bebop_df"].index.values
        head_t = bag_df_dict["head_df"].index.values
        odom_t = bag_df_dict["odom_df"].index.values
        data_vec = []
        min_ = bag_start_cut[f[:-4]]
        max_ = bag_end_cut[f[:-4]]
        for i in tqdm.tqdm(range(min_, max_), desc="processing data " + str(data_id)):
            b_id = find_nearest(bebop_t, camera_t[i])
            h_id = find_nearest(head_t, camera_t[i])
            odom_id = find_nearest(odom_t, camera_t[i])
            head_pose = bag_df_dict["head_df"].iloc[h_id]
            bebop_pose = bag_df_dict["bebop_df"].iloc[b_id]
            odom_info = bag_df_dict["odom_df"].iloc[odom_id]
            img = bag_df_dict["camera_df"].iloc[i].values[0]
            b_t_h = change_frame_reference(bebop_pose, head_pose)
            vel_x = odom_info.x_vel
            vel_y = odom_info.y_vel

            quaternion_bebop = bebop_pose[['b_rot_x', 'b_rot_y', 'b_rot_z', 'b_rot_w']].values
            _, _, bebop_yaw = quat_to_eul(quaternion_bebop)
            quaternion_head = head_pose[['h_rot_x', 'h_rot_y', 'h_rot_z', 'h_rot_w']].values
            _, _, head_yaw = quat_to_eul(quaternion_head)

            relative_yaw = (head_yaw - bebop_yaw - np.pi)
            if relative_yaw < -np.pi:
                relative_yaw += 2 * np.pi

            target_position = b_t_h[:-1, -1:].T[0]
            target = (target_position[0], target_position[1], target_position[2], relative_yaw)
            data_vec.append((img, target, np.asarray([vel_x, vel_y])))
        return data_vec

    elif model_type == "v2":
        camera_t = bag_df_dict["camera_df"].index.values
        bebop_t = bag_df_dict["bebop_df"].index.values
        head_t = bag_df_dict["head_df"].index.values
        odom_t = bag_df_dict["odom_df"].index.values
        bebop_twist_t = bag_df_dict["bebop_twist_df"].index.values

        max_ = bag_end_cut[f[:-4]]
        min_ = bag_start_cut[f[:-4]]
        data_vec = []
        d_ctrl = Controller()
        for i in tqdm.tqdm(range(min_, max_), desc="processing data " + str(data_id)):
            b_id = find_nearest(bebop_t, camera_t[i])
            h_id = find_nearest(head_t, camera_t[i])
            odom_id = find_nearest(odom_t, camera_t[i])
            bebop_twist_id = find_nearest(bebop_twist_t, camera_t[i])
            head_pose = bag_df_dict["head_df"].iloc[h_id]
            bebop_pose = bag_df_dict["bebop_df"].iloc[b_id]
            odom_info = bag_df_dict["odom_df"].iloc[odom_id]
            bebop_twist = bag_df_dict["bebop_twist_df"].iloc[bebop_twist_id]
            img = bag_df_dict["camera_df"].iloc[i].values[0]
            b_t_h = change_frame_reference(bebop_pose, head_pose)
            vel_x = odom_info.x_vel
            vel_y = odom_info.y_vel

            quaternion_bebop = bebop_pose[['b_rot_x', 'b_rot_y', 'b_rot_z', 'b_rot_w']].values
            _, _, bebop_yaw = quat_to_eul(quaternion_bebop)
            quaternion_head = head_pose[['h_rot_x', 'h_rot_y', 'h_rot_z', 'h_rot_w']].values
            _, _, head_yaw = quat_to_eul(quaternion_head)

            relative_yaw = (head_yaw - bebop_yaw - np.pi)
            if relative_yaw < -np.pi:
                relative_yaw += 2 * np.pi

            target_position = b_t_h[:-1, -1:].T[0]
            new_bebop_twist = change_frame_reference_twist(bebop_pose, bebop_twist)
            new_bebop_twist = new_bebop_twist[:-1, -1:].T[0]

            v_drone = [new_bebop_twist[0], new_bebop_twist[1], 0]
            target = d_ctrl.new_controller(target_position, relative_yaw, v_drone, delay=0.0)
            data_vec.append((img, np.asarray([vel_x, vel_y]), target))
        return data_vec
    elif model_type == "v3":
        camera_t = bag_df_dict["camera_df"].index.values
        odom_t = bag_df_dict["odom_df"].index.values
        bebop_t = bag_df_dict["bebop_df"].index.values
        head_t = bag_df_dict["head_df"].index.values
        bebop_twist_t = bag_df_dict["bebop_twist_df"].index.values

        max_ = bag_end_cut[f[:-4]]
        min_ = bag_start_cut[f[:-4]]
        data_vec = []
        d_ctrl = Controller()
        for i in tqdm.tqdm(range(min_, max_), desc="processing data " + str(data_id)):
            odom_id = find_nearest(odom_t, camera_t[i])
            b_id = find_nearest(bebop_t, camera_t[i])
            h_id = find_nearest(head_t, camera_t[i])
            bebop_twist_id = find_nearest(bebop_twist_t, camera_t[i])
            bebop_twist = bag_df_dict["bebop_twist_df"].iloc[bebop_twist_id]
            odom_info = bag_df_dict["odom_df"].iloc[odom_id]
            head_pose = bag_df_dict["head_df"].iloc[h_id]
            bebop_pose = bag_df_dict["bebop_df"].iloc[b_id]
            img = bag_df_dict["camera_df"].iloc[i].values[0]
            vel_x = odom_info.x_vel
            vel_y = odom_info.y_vel
            b_t_h = change_frame_reference(bebop_pose, head_pose)

            quaternion_bebop = bebop_pose[['b_rot_x', 'b_rot_y', 'b_rot_z', 'b_rot_w']].values
            _, _, bebop_yaw = quat_to_eul(quaternion_bebop)
            quaternion_head = head_pose[['h_rot_x', 'h_rot_y', 'h_rot_z', 'h_rot_w']].values
            _, _, head_yaw = quat_to_eul(quaternion_head)

            relative_yaw = (head_yaw - bebop_yaw - np.pi)
            if relative_yaw < -np.pi:
                relative_yaw += 2 * np.pi

            target_position = b_t_h[:-1, -1:].T[0]
            new_bebop_twist = change_frame_reference_twist(bebop_pose, bebop_twist)
            new_bebop_twist = new_bebop_twist[:-1, -1:].T[0]

            v_drone = [new_bebop_twist[0], new_bebop_twist[1], 0]
            target = d_ctrl.new_controller(target_position, relative_yaw, v_drone, delay=0.0)
            rel_head = (target_position[0], target_position[1], target_position[2], relative_yaw)
            data_vec.append((img, rel_head, np.asarray([vel_x, vel_y]), target))
        return data_vec
    else:
        print('Error in video_selected in processing')
        sys.exit(1)


def inner_method_temp_name(f, model_type):
    """
        Core method used to transforms and saves a bag file into a .pickle dataset file
    Args:
        f: file name e.g. "7.bag"
        model_type: type of model that will use dataset
    """
    path = bag_file_path[f[:-4]]
    print("\nreading bag: " + str(f))
    datacr = DatasetCreator()
    with rosbag.Bag(path + f) as bag:
        bag_df_dict = get_bag_data_pandas(bag, model_type=model_type)
    data_vec = processing(bag_df_dict=bag_df_dict, data_id=f[:-4], f=f, model_type=model_type)
    datacr.generate_data(data_vec=data_vec)
    f__pickle_ = f[:-4] + ".pickle"
    datacr.save_dataset(flag_train="cross", title=f__pickle_, model_type=model_type)
    print("\nCompleted pickle " + str(f[:-4]))


def bag_to_pickle_middle(f):
    """
        Core method used to transforms and saves a bag file into a .pickle dataset file
    Args:
        f: file name e.g. "7.bag"
    """
    inner_method_temp_name(f, "v1")


def bag_to_pickle_controller(f):
    """
        Core method used to transforms and saves a bag file into a .pickle dataset file
    Args:
        f: file name e.g. "7.bag"
    """
    inner_method_temp_name(f, "v2")


def bag_to_pickle_vers3(f):
    """
        Core method used to transforms and saves a bag file into a .pickle dataset file
    Args:
        f: file name e.g. "7.bag"
    """
    inner_method_temp_name(f, "v3")

def main():
    """
        Using user input from console select which functionaly execute:
            - create single dataset (Single threaded script)
            - create crossvalidation dataset (Multi threaded script, high CPU usage.
                                                Can run single thread for debugging)
    Returns:
        None in case of errors
    """
    scelta = raw_input("experiment_version:\n    -v1[1]\n    -v2[2]\n    -v3[3]\n")
    if scelta == "1":
        model_type = "v1"
        path = "../bagfiles/train/"
        print("creating Train set")
        files = [f for f in os.listdir(path) if f[-4:] == '.bag']
        if not files:
            print('No bag files found!')
            return None
        datacr_train = DatasetCreator()
        for f in files:
            path = bag_file_path[f[:-4]]
            print("\nreading bag: " + str(f))
            with rosbag.Bag(path + f) as bag:
                bag_df_dict = get_bag_data_pandas(bag, model_type=model_type)
            data_vec = processing(bag_df_dict=bag_df_dict, data_id=f[:-4], f=f, model_type=model_type)
            datacr_train.generate_data(data_vec=data_vec)
        datacr_train.save_dataset(flag_train="train", model_type=model_type)

        path = "../bagfiles/validation/"
        print("creating test set")
        files = [f for f in os.listdir(path) if f[-4:] == '.bag']
        if not files:
            print('No bag files found!')
            return None
        datacr_val = DatasetCreator()
        for f in files:
            path = bag_file_path[f[:-4]]
            print("\nreading bag: " + str(f))
            with rosbag.Bag(path + f) as bag:
                bag_df_dict = get_bag_data_pandas(bag, model_type=model_type)
            data_vec = processing(bag_df_dict=bag_df_dict, data_id=f[:-4], f=f, model_type=model_type)
            datacr_val.generate_data(data_vec=data_vec)
        datacr_val.save_dataset(flag_train="test", model_type=model_type)
    elif scelta == "2":
        model_type = "v2"
        path = "../bagfiles/train/"
        files = [f for f in os.listdir(path) if f[-4:] == '.bag']
        if not files:
            print('No bag files found!')
            return None
        datacr_train = DatasetCreator()
        for f in files:
            path = bag_file_path[f[:-4]]
            print("\nreading bag: " + str(f))
            with rosbag.Bag(path + f) as bag:
                bag_df_dict = get_bag_data_pandas(bag, model_type=model_type)
            data_vec = processing(bag_df_dict=bag_df_dict, data_id=f[:-4], f=f, model_type=model_type)
            datacr_train.generate_data(data_vec=data_vec)
        datacr_train.save_dataset(flag_train="train", model_type=model_type)

        path = "../bagfiles/validation/"
        files = [f for f in os.listdir(path) if f[-4:] == '.bag']
        if not files:
            print('No bag files found!')
            return None
        datacr_val = DatasetCreator()
        for f in files:
            path = bag_file_path[f[:-4]]
            print("\nreading bag: " + str(f))
            with rosbag.Bag(path + f) as bag:
                bag_df_dict = get_bag_data_pandas(bag, model_type=model_type)
            data_vec = processing(bag_df_dict=bag_df_dict, data_id=f[:-4], f=f, model_type=model_type)
            datacr_val.generate_data(data_vec=data_vec)
        datacr_val.save_dataset(flag_train="test", model_type=model_type)
    elif scelta == "3":
        model_type = "v3"
        scelta2 = raw_input("single or cross:[s/c]")
        if scelta2 == "s":
            path = "../bagfiles/train/"
            files = [f for f in os.listdir(path) if f[-4:] == '.bag']
            if not files:
                print('No bag files found!')
                return None
            datacr_train = DatasetCreator()
            for f in files:
                path = bag_file_path[f[:-4]]
                print("\nreading bag: " + str(f))
                with rosbag.Bag(path + f) as bag:
                    bag_df_dict = get_bag_data_pandas(bag, model_type=model_type)
                data_vec = processing(bag_df_dict=bag_df_dict, data_id=f[:-4], f=f, model_type=model_type)
                datacr_train.generate_data(data_vec=data_vec)
            datacr_train.save_dataset(flag_train="train", model_type=model_type)

            path = "../bagfiles/validation/"
            files = [f for f in os.listdir(path) if f[-4:] == '.bag']
            if not files:
                print('No bag files found!')
                return None
            datacr_val = DatasetCreator()
            for f in files:
                path = bag_file_path[f[:-4]]
                print("\nreading bag: " + str(f))
                with rosbag.Bag(path + f) as bag:
                    bag_df_dict = get_bag_data_pandas(bag, model_type=model_type)
                data_vec = processing(bag_df_dict=bag_df_dict, data_id=f[:-4], f=f, model_type=model_type)
                datacr_val.generate_data(data_vec=data_vec)
            datacr_val.save_dataset(flag_train="test", model_type=model_type)

        elif scelta2 == "c":
            path1 = "../bagfiles/train/"
            path2 = "../bagfiles/validation/"

            files1 = [f for f in os.listdir(path1) if f[-4:] == '.bag']
            if not files1:
                print('No bag files found!')
                return None
            files2 = [f for f in os.listdir(path2) if f[-4:] == '.bag']
            if not files2:
                print('No bag files found!')
                return None
            files = []
            for f_ in files1:
                files.append(f_)
            for f_ in files2:
                files.append(f_)

            scelta_3 = raw_input("Singlethread or multi-threaded:[s/m]")
            if scelta_3 == 's':
                for f in files:
                    bag_to_pickle_vers3(f)
            else:
                pool = Pool(processes=4)
                pool.map(bag_to_pickle_vers3, files[:])
                pool.close()
                pool.join()
        else:
            print('Error in selection')
            sys.exit(1)
    else:
        print('Error in selection')
        sys.exit(1)


if __name__ == "__main__":
    main()

