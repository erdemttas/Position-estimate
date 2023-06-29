import cv2
import mediapipe as mp   #mediapipe, yapay zeka tabanlı algoritmalar ve araçlar sunar
import time

mpPose = mp.solutions.pose   #mediapipe pose modelini kullanabilmek için mpPose nesnesini oluşturduk.
pose = mpPose.Pose()   #Pose modelinin bir örneğini pose olarak oluşturduk.

mpDraw = mp.solutions.drawing_utils   #bu satır modelin eklem yerlerinin çizilmesini sağlar.

cap = cv2.VideoCapture("video4.mp4")   #videoları içeri aktardık.

pTime = 0   #fps hesaplanması için previous time adlı değişken oluşturduk.
while True:
    success, img = cap.read()   #gelen her video karesinin başarılı bir şekilde gelip gelmediğini kontrol eder ve "success" değişkenine True yada False değeri döndürür. Başarılı bir şekilde gelen resim karelerini img parametresine aktarır.
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   #OpenCV renkleri BGR(blue-green-red) olarak algılar, Gerçek hayatta renkler RGB dir. Bu yüzden BGR dan RGB ye renk dönüşümü yaparız.
    results = pose.process(imgRGB)   #Renklerini ayarladığımız video karesini pose modeline gönderir ve değerleri alırız.
    #print(results.pose_landmarks)   #görüntüdeki eklem noktalarının çıktısını yazdırır.
    
    if results.pose_landmarks:   #eklem noktaları videonun içerisinde varsa aşağıdaki işlemler gerçekleşsin.
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)   #eklem noktalarını ve bağlantıları çizmemizi sağlar.
        
        for id, lm in enumerate(results.pose_landmarks.landmark):   #eklemlerin id ve kordinat bilgileri döner.
            h, w, _ = img.shape   #video karesinin yükseklik, genişlik ve renk kanalı değerlerini alırız ancak renk kanalına ihtiyacımız olmadığı için _ boş geçeriz.
            cx, cy = int(lm.x*w), int(lm.y*h)   #eklem noktalarının kordinat bilgilerini tespit ederiz.
            
            if id==13:   # 13 id li eklemin mavi yuvarlakla işaretlenmesi.
                cv2.circle(img, (cx,cy), 8, (255,0,0), cv2.FILLED)
    
    
    cTime = time.time()   #şimdiki zamandan önceki zamanı çıkarıp 1'e bölersek fps'i bulmuş oluruz.
    fps = 1/(cTime - pTime)
    pTime = cTime    
    
    cv2.putText(img, "FPS: "+str(int(fps)), (10,65), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)   #fps bilgisinin görüntüye yazılması.
    
    cv2.imshow("img",img)   # görüntünün gösterilmesi
    cv2.waitKey(5)   #5 milisaniye bekleyerek video karelerinin ekrana gelmesini sağlar.
    
    
    
    
    
    
    
    
    
    
    
    