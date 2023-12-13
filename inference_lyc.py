import argparse
import numpy as np
import open3d as o3d
import pickle

def inference_single(point_info, results_folder):

        shape_id = point_info['shape_id']
        points_coordinates = point_info['shape']
        affordance_label = point_info['gt_mask']
        affordance_pred = point_info['pred']
        iou = point_info['iou']
        object = point_info['cls']
        affordance_type = point_info['aff']

        gt_point = o3d.geometry.PointCloud()
        gt_point.points = o3d.utility.Vector3dVector(points_coordinates)

        pred_point = o3d.geometry.PointCloud()
        pred_point.points = o3d.utility.Vector3dVector(points_coordinates)

        color = np.zeros((2048,3))
        reference_color = np.array([255, 0, 0])
        back_color = np.array([190, 190, 190])

        for i, point_affordacne in enumerate(affordance_label):
            scale_i = point_affordacne
            color[i] = (reference_color-back_color) * scale_i + back_color
        gt_point.colors = o3d.utility.Vector3dVector(color.astype(np.float64) / 255.0)

        pred_color = np.zeros((2048,3))

        for i, aff_pred in enumerate(affordance_pred):
            scale_i = aff_pred
            pred_color[i] = (reference_color-back_color) * scale_i + back_color
        pred_point.colors = o3d.utility.Vector3dVector(pred_color.astype(np.float64) / 255.0)
        pred_point.translate((2, 0, 0), relative=True)

        GT_file = results_folder + object + '_' + affordance_type + '_' + str(round(iou,3)) + '_'+ shape_id + '_GT' + '.ply'
        pred_file = results_folder + object + '_' + affordance_type + '_' + str(round(iou,3)) + '_'+ shape_id +  '_Pred' + '.ply'

        o3d.visualization.draw_geometries([gt_point, pred_point], window_name='GT point', width=600, height=600)

        o3d.io.write_point_cloud(pred_file, pred_point)
        o3d.io.write_point_cloud(GT_file, gt_point)   

        return GT_file, pred_file

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--point_path', type=str, default='/storage_fast/ycli/3d_affordance/visualizer/data/top5-objaff.pkl', help='test point path')
    parser.add_argument('--results_path', type=str, default='/storage_fast/ycli/3d_affordance/visualizer/ply_file/', help='save Demo path')

    opt = parser.parse_args()

    with open(opt.point_path, 'rb') as file:
        top_points = pickle.load(file)[:2]
    
    with open('/storage_fast/ycli/3d_affordance/visualizer/ply_file/path2all.text', 'w') as file:
        for points_info in top_points:
            GT_file, pred_file = inference_single(points_info, opt.results_path)
            file.write(GT_file + '\n')
            file.write(pred_file + '\n')
