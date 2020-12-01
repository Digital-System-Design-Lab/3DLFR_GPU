# 3DLFR_GPU
GPU-based 3D Light Field Rendering Accelerator

LF_Renderer ----+---- LRU_Cache --+-- Slice ---- SliceID
                |                 |
                |                 +-- DeviceMemoryManger
                |
                |
                +---- LFU_Window ---- LFU ---- Interlaced_LF
                
                

* SliceID     : LF Number, Image number, Slice number를 포함하는 구조체. 해시맵 검색에 사용됨.
* Slice       : Slice ID와 RGB 데이터의 주소, prev/next 포인터를 가짐. 리스트의 원소.
* LRU_Cache   : 해시맵과 Doubly-linked list<Slice>로 구성된 LRU 캐시. put 함수가 캐싱 동작을 함.
                  1) put 함수는 Slice ID와 LF 데이터의 주소를 인자로 받고, Slice ID로 해시맵 검색을 수행한다.
                  2-1) Cache miss일 때는 Slice를 host memory에서 동적할당 후 인자로 받은 ID와 LF 데이터를 Slice에 넣고 리스트에 추가한다.
                  2-2) 리스트에 추가된 슬라이스의 데이터 필드를 device memory에 복사한다.
                  2-3) 추가된 슬라이스의 device memory 주소를 h_devPtr_hashmap[]에 추가한다. 
                  3) 렌더링 하기 직전에 d_devPtr_hashmap[] <- h_devPtr_hashmap[]의 데이터 복사 (동기화)가 이루어진다.
                  * h_devPtr_hashmap[], d_devPtr_hashmap[] 배열은 device memory에 저장된 슬라이스의 주소를 담는 공간임.
                    (prefix 'h_'는 host, 'd_'는 device에서 할당된 공간을 의미)
                    h_devPtr_hashmap[]은 device memory의 주소를 저장할 뿐 Host memory의 heap에 생성된 공간이기 때문에 GPU 커널에서 접근할 수 없음
                    이러한 이유로 3)의 동기화 과정이 필요함.
                    GPU 커널은 PosX, PosY를 토대로 LF number/Image number/Slice number를 계산하며, 이 값으로 d_devPtr_hashmap에 접근해 slice의 픽셀에 접근함.

* LFU__Window : Disk로부터 LF 파일을 Host memory로 읽어온다.
