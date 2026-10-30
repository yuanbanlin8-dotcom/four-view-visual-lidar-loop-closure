from pathlib import Path
import os,signal,socket,subprocess,time,xmlrpc.client
root=Path(__file__).resolve().parents[1];env=dict(os.environ)
reports=root/'reports';reports.mkdir(parents=True,exist_ok=True)
env.update(ROS_MASTER_URI='http://127.0.0.1:11329',ROS_HOSTNAME='127.0.0.1',QT_QPA_PLATFORM='offscreen')
with socket.socket() as s:
    if s.connect_ex(('127.0.0.1',11329))==0: raise RuntimeError('Test master port already occupied')
with (reports/'synthetic-roscore.log').open('w') as log:
    master=subprocess.Popen(['roscore','-p','11329'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        socket.setdefaulttimeout(0.5)
        ready=False
        for _ in range(100):
            try:
                if xmlrpc.client.ServerProxy(env['ROS_MASTER_URI']).getPid('/smoke')[0]==1:
                    ready=True;break
            except Exception: time.sleep(0.1)
        if not ready: raise RuntimeError('Test ROS master failed to start')
        subprocess.run(['rosparam','load',str(root/'tests/synthetic_only.yaml')],env=env,check=True)
        subprocess.run(['rosparam','load',str(root/'src/Omni-LIVO/config/seeker_raw_cam.yaml'),'/laserMapping'],env=env,check=True)
        with (reports/'backend-smoke.log').open('w') as out:
            backend_result=subprocess.run([str(root/'devel/lib/omni_livo/raw_backend_smoke')],env=env,stdout=out,stderr=subprocess.STDOUT,timeout=120)
        subprocess.run(['rosparam','set','/pcd_save/pcd_save_en','true'],env=env,check=True)
        subprocess.run(['rosparam','set','/pcd_save/interval','1'],env=env,check=True)
        subprocess.run(['rosparam','set','/pcd_save/filter_size_pcd','0.05'],env=env,check=True)
        with (reports/'pcd-writer-smoke.log').open('w') as out:
            pcd_result=subprocess.run([str(root/'devel/lib/omni_livo/pcd_writer_smoke')],env=env,stdout=out,stderr=subprocess.STDOUT,timeout=120)
        result_code=backend_result.returncode or pcd_result.returncode
        (reports/'backend-smoke.exit').write_text(str(result_code))
        print('SMOKE_EXIT='+str(backend_result.returncode))
        print('PCD_WRITER_SMOKE_EXIT='+str(pcd_result.returncode))
    finally:
        os.killpg(master.pid,signal.SIGTERM)
        try: master.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(master.pid,signal.SIGKILL);master.wait()

raise SystemExit(0 if result_code==0 else 1)
