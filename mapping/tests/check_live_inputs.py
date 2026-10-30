import json,time,threading
from pathlib import Path
import rospy,yaml
from sensor_msgs.msg import Image
root=Path('/home/nuc/omni_livo_fisheye_ws')
data=yaml.safe_load(Path('/home/nuc/seeker_ws/calibration/factory_20260910.yaml').read_text())
rospy.init_node('seeker_raw_input_check',anonymous=True)
lock=threading.Lock(); stats={}; stamps={}
def cb(msg,i):
    with lock:
        key='cam'+str(i)
        stats[key]=dict(topic=data[key]['rostopic'],width=msg.width,height=msg.height,encoding=msg.encoding,step=msg.step)
        stamps.setdefault(key,[]).append(msg.header.stamp.to_nsec())
subs=[rospy.Subscriber(data['cam'+str(i)]['rostopic'],Image,cb,i,queue_size=1,buff_size=8*1024*1024) for i in range(4)]
end=time.monotonic()+5
while time.monotonic()<end and not rospy.is_shutdown(): time.sleep(0.05)
with lock:
    for key,item in stats.items():
        ts=stamps[key];item['frames']=len(ts)
        item['estimated_hz']=(len(ts)-1)*1e9/(ts[-1]-ts[0]) if len(ts)>1 and ts[-1]>ts[0] else None
        item['strictly_increasing_timestamps']=all(b>a for a,b in zip(ts,ts[1:]))
        diffs=[(b-a)/1e6 for a,b in zip(ts,ts[1:])]
        item['duplicate_stamps']=sum(d==0 for d in diffs)
        item['backward_stamps']=sum(d<0 for d in diffs)
        item['min_delta_ms']=min(diffs) if diffs else None
        item['max_delta_ms']=max(diffs) if diffs else None
        item['matches_factory_resolution']=[item['width'],item['height']]==data[key]['resolution']
    shared=len(set.intersection(*(set(stamps.get('cam'+str(i),[])) for i in range(4))))
    report=dict(cameras=stats,shared_exact_timestamps=shared,note='Camera input check only; not a LiDAR synchronization or calibration validation.')
    (root/'reports/live-inputs.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False))
rospy.signal_shutdown('done')
