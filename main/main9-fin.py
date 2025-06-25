import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ultralytics import YOLO
import cv2
import os
import threading
import time

class YOLODetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Detection GUI")
        self.root.geometry("800x600")
        
        # Biến để lưu đường dẫn
        self.file_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.image_output_path = tk.StringVar()
        self.directory_path = tk.StringVar()  # Thêm biến cho đường dẫn thư mục
        self.detection_mode = tk.StringVar(value="image")
        self.confidence = tk.DoubleVar(value=0.5)
        
        # Biến cho chế độ duyệt thư mục
        self.current_images = []  # Danh sách các file ảnh trong thư mục
        self.current_image_index = -1  # Index của ảnh hiện tại
        self.current_output_image = None  # Lưu ảnh đã xử lý hiện tại
        
        # Tạo và cấu hình GUI
        self.setup_gui()
        
        # Khởi tạo model YOLO
        self.model = None
        try:
            self.model = YOLO(r'E:\Code\TGMT\BTL\main\runs6(150)\detect\train\weights\best.pt')
        except Exception as e:
            messagebox.showerror("Error", f"Không thể tải model YOLO: {str(e)}")
        
        # Bind phím tắt
        self.root.bind('<Left>', self.prev_image)
        self.root.bind('<Right>', self.next_image)
        self.root.bind('s', self.save_current_image)
    
    def setup_gui(self):
        # Frame chính
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Chọn chế độ
        mode_frame = ttk.LabelFrame(main_frame, text="Chọn chế độ", padding="5")
        mode_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Radiobutton(mode_frame, text="Ảnh", variable=self.detection_mode, 
                       value="image", command=self.update_gui).grid(row=0, column=0, padx=5)
        ttk.Radiobutton(mode_frame, text="Video", variable=self.detection_mode, 
                       value="video", command=self.update_gui).grid(row=0, column=1, padx=5)
        ttk.Radiobutton(mode_frame, text="Thư mục ảnh", variable=self.detection_mode,
                       value="directory", command=self.update_gui).grid(row=0, column=2, padx=5)
        
        # Frame chọn file/thư mục
        file_frame = ttk.LabelFrame(main_frame, text="Chọn file/thư mục", padding="5")
        file_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        self.file_label = ttk.Label(file_frame, text="File path:")
        self.file_label.grid(row=0, column=0, padx=5)
        self.file_entry = ttk.Entry(file_frame, textvariable=self.file_path, width=50)
        self.file_entry.grid(row=0, column=1, padx=5)
        self.browse_button = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        self.browse_button.grid(row=0, column=2, padx=5)
        
        # Frame thông tin điều hướng (cho chế độ thư mục)
        self.nav_frame = ttk.LabelFrame(main_frame, text="Điều hướng", padding="5")
        self.nav_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Label(self.nav_frame, text="Sử dụng:").grid(row=0, column=0, padx=5)
        ttk.Label(self.nav_frame, text="← → : Ảnh trước/sau").grid(row=0, column=1, padx=5)
        ttk.Label(self.nav_frame, text="S : Lưu ảnh").grid(row=0, column=2, padx=5)
        self.image_counter = ttk.Label(self.nav_frame, text="Ảnh: 0/0")
        self.image_counter.grid(row=0, column=3, padx=5)
        
        # Frame output cho ảnh
        self.image_output_frame = ttk.LabelFrame(main_frame, text="Output Ảnh", padding="5")
        self.image_output_frame.grid(row=3, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Label(self.image_output_frame, text="Output path:").grid(row=0, column=0, padx=5)
        ttk.Entry(self.image_output_frame, textvariable=self.image_output_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(self.image_output_frame, text="Browse", command=self.browse_image_output).grid(row=0, column=2, padx=5)
        
        # Frame output cho video
        self.video_output_frame = ttk.LabelFrame(main_frame, text="Output Video", padding="5")
        self.video_output_frame.grid(row=4, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Label(self.video_output_frame, text="Output path:").grid(row=0, column=0, padx=5)
        ttk.Entry(self.video_output_frame, textvariable=self.output_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(self.video_output_frame, text="Browse", command=self.browse_output).grid(row=0, column=2, padx=5)
        
        # Frame cài đặt
        settings_frame = ttk.LabelFrame(main_frame, text="Cài đặt", padding="5")
        settings_frame.grid(row=5, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Label(settings_frame, text="Confidence threshold:").grid(row=0, column=0, padx=5)
        confidence_scale = ttk.Scale(settings_frame, from_=0.0, to=1.0, 
                                   variable=self.confidence, orient=tk.HORIZONTAL)
        confidence_scale.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        
        self.confidence_label = ttk.Label(settings_frame, text=f"{self.confidence.get():.2f}")
        self.confidence_label.grid(row=0, column=2, padx=5)
        
        self.confidence.trace_add('write', self.update_confidence_label)

        # Nút thực thi
        ttk.Button(main_frame, text="Start Detection", command=self.start_detection).grid(row=6, column=0, columnspan=2, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Cập nhật GUI ban đầu
        self.update_gui()
    
    def update_gui(self):
        mode = self.detection_mode.get()
        # Cập nhật label và nút browse
        if mode == "directory":
            self.file_label.config(text="Directory path:")
            self.browse_button.config(command=self.browse_directory)
            self.nav_frame.grid()
        else:
            self.file_label.config(text="File path:")
            self.browse_button.config(command=self.browse_file)
            self.nav_frame.grid_remove()
        
        # Hiển thị/ẩn các frame output
        if mode == "video":
            self.video_output_frame.grid()
            self.image_output_frame.grid_remove()
        elif mode == "image":
            self.video_output_frame.grid_remove()
            self.image_output_frame.grid()
        else:  # directory mode
            self.video_output_frame.grid_remove()
            self.image_output_frame.grid_remove()
            # self.image_output_frame.grid()
    
    def browse_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.file_path.set(directory)
            self.load_directory_images(directory)
    
    def load_directory_images(self, directory):
        # Lấy danh sách các file ảnh trong thư mục
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.jfif')
        self.current_images = [
            os.path.join(directory, f) for f in os.listdir(directory)
            if f.lower().endswith(image_extensions)
        ]
        self.current_images.sort()
        
        if self.current_images:
            self.current_image_index = 0
            self.update_image_counter()
            self.process_current_image()
        else:
            messagebox.showwarning("Warning", "Không tìm thấy ảnh trong thư mục")
    
    def update_image_counter(self):
        if self.current_images:
            self.image_counter.config(
                text=f"Ảnh: {self.current_image_index + 1}/{len(self.current_images)}"
            )
    
    def prev_image(self, event=None):
        if self.current_images and self.current_image_index > 0:
            self.current_image_index -= 1
            self.update_image_counter()
            self.process_current_image()
    
    def next_image(self, event=None):
        if self.current_images and self.current_image_index < len(self.current_images) - 1:
            self.current_image_index += 1
            self.update_image_counter()
            self.process_current_image()
    
    def process_current_image(self):
        if 0 <= self.current_image_index < len(self.current_images):
            image_path = self.current_images[self.current_image_index]
            try:
                # Đọc ảnh
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError("Không thể đọc ảnh")
                
                # Thực hiện dự đoán
                results = self.model(image)[0]
                
                # Tạo bản sao của ảnh để vẽ kết quả
                output_image = image.copy()
                
                # Vẽ kết quả
                for result in results.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = result
                    
                    if score > self.confidence.get():
                        class_name = results.names[int(class_id)]
                        cv2.rectangle(output_image, 
                                    (int(x1), int(y1)), 
                                    (int(x2), int(y2)), 
                                    (0, 255, 0), 
                                    2)
                        label = f"{class_name}: {score:.2f}"
                        cv2.putText(output_image, 
                                  label, 
                                  (int(x1), int(y1 - 10)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, 
                                  (0, 255, 0), 
                                  2)
                
                # Lưu ảnh đã xử lý vào biến
                self.current_output_image = output_image
                
                # Hiển thị ảnh
                cv2.imshow("Detection Result", output_image)
                cv2.waitKey(1)  # Sử dụng waitKey(1) thay vì waitKey(0)
                
            except Exception as e:
                messagebox.showerror("Error", str(e))
                self.status_var.set("Lỗi khi xử lý ảnh")
    
    def save_current_image(self, event=None):
        if self.current_output_image is not None:
            try:
                # Tạo tên file mặc định
                if self.current_image_index >= 0 and self.current_image_index < len(self.current_images):
                    original_path = self.current_images[self.current_image_index]
                    directory = os.path.dirname(original_path)
                    basename = os.path.basename(original_path)
                    name, ext = os.path.splitext(basename)
                    default_output = os.path.join(directory, f"detected_{name}{ext}")
                    
                    # Hiển thị dialog save file
                    filename = filedialog.asksaveasfilename(
                        initialfile=f"detected_{name}{ext}",
                        defaultextension=ext,
                        filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
                    )
                    
                    if filename:
                        cv2.imwrite(filename, self.current_output_image)
                        self.status_var.set(f"Đã lưu ảnh tại: {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Lỗi khi lưu ảnh: {str(e)}")
    
    def update_confidence_label(self, *args):
        self.confidence_label.config(text=f"{self.confidence.get():.2f}")
    
    def browse_file(self):
        filetypes = []
        if self.detection_mode.get() == "image":
            filetypes = [("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.jfif")]
        else:
            filetypes = [("Video files", "*.mp4 *.avi *.mov *.mkv")]
        
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            self.file_path.set(filename)
            directory = os.path.dirname(filename)
            basename = os.path.basename(filename)
            
            if self.detection_mode.get() == "video":
                output_name = f"detected_{basename}"
                self.output_path.set(os.path.join(directory, output_name))
            else:
                name, ext = os.path.splitext(basename)
                output_name = f"detected_{name}{ext}"
                self.image_output_path.set(os.path.join(directory, output_name))
    
    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi")]
        )
        if filename:
            self.output_path.set(filename)
    
    def browse_image_output(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if filename:
            self.image_output_path.set(filename)
    
    def process_image(self):
        try:
            # Đọc ảnh
            image = cv2.imread(self.file_path.get())
            if image is None:
                raise ValueError("Không thể đọc ảnh")
            
            # Thực hiện dự đoán
            results = self.model(image)[0]
            
            # Tạo bản sao của ảnh để vẽ kết quả
            output_image = image.copy()
            
            # Vẽ kết quả
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                
                if score > self.confidence.get():
                    class_name = results.names[int(class_id)]
                    cv2.rectangle(output_image, 
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)), 
                                (0, 255, 0), 
                                2)
                    label = f"{class_name}: {score:.2f}"
                    cv2.putText(output_image, 
                              label, 
                              (int(x1), int(y1 - 10)), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, 
                              (0, 255, 0), 
                              2)
            
            # Lưu ảnh nếu có đường dẫn output
            if self.image_output_path.get():
                cv2.imwrite(self.image_output_path.get(), output_image)
                self.status_var.set(f"Đã lưu ảnh tại: {self.image_output_path.get()}")
            
            # Hiển thị ảnh
            cv2.imshow("Detection Result", output_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Lỗi khi xử lý ảnh")
    
    def process_video(self):
        try:
            cap = cv2.VideoCapture(self.file_path.get())
            if not cap.isOpened():
                raise ValueError("Không thể mở video")
            
            # Lấy thông tin video
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Khởi tạo video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path.get(), fourcc, fps, (frame_width, frame_height))
            
            frame_count = 0
            prev_time = time.time()
            fps_counter = 0
            fps_to_display = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                fps_counter += 1
                if time.time() - prev_time > 1.0:
                    fps_to_display = fps_counter
                    fps_counter = 0
                    prev_time = time.time()
                
                frame_count += 1
                self.status_var.set(f"Đang xử lý frame {frame_count}")
                self.root.update()
                
                # Thực hiện dự đoán
                results = self.model(frame)[0]
                
                # Vẽ kết quả
                for result in results.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = result
                    
                    if score > self.confidence.get():
                        class_name = results.names[int(class_id)]
                        cv2.rectangle(frame, 
                                    (int(x1), int(y1)), 
                                    (int(x2), int(y2)), 
                                    (0, 255, 0), 
                                    2)
                        label = f"{class_name}: {score:.2f}"
                        cv2.putText(frame, 
                                  label, 
                                  (int(x1), int(y1 - 10)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, 
                                  (0, 255, 0), 
                                  2)
                
                # Hiển thị FPS
                cv2.putText(frame, 
                           f"FPS: {fps_to_display}", 
                           (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           1, 
                           (0, 255, 0), 
                           2)
                
                # Lưu frame
                out.write(frame)
                
                # Hiển thị frame
                cv2.imshow("Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            self.status_var.set(f"Hoàn thành xử lý video. Đã xử lý {frame_count} frames")
        
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Lỗi khi xử lý video")
    
    def start_detection(self):
        if not self.file_path.get():
            messagebox.showerror("Error", "Vui lòng chọn file")
            return
        
        # if self.detection_mode.get() == "video" and not self.output_path.get():
        #     messagebox.showerror("Error", "Vui lòng chọn đường dẫn output cho video")
        #     return
        
        # if self.detection_mode.get() == "image" and not self.image_output_path.get():
        #     messagebox.showerror("Error", "Vui lòng chọn đường dẫn lưu ảnh")
        #     return
        
        # Bắt đầu xử lý
        if self.detection_mode.get() == "image":
            threading.Thread(target=self.process_image, daemon=True).start()
        else:
            threading.Thread(target=self.process_video, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLODetectionGUI(root)
    root.mainloop()