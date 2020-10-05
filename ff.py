import time

timer=time.time()
while True:
    if (time.time()-timer)>3:
        print('ok')
        timer=time.time()
