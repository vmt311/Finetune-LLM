# In BookLoader.py

from tqdm import tqdm # Import thư viện tqdm để tạo thanh tiến trình (progress bar)
from concurrent.futures import ProcessPoolExecutor # Import ProcessPoolExecutor để xử lý song song trên nhiều tiến trình
from BookItems import BookItem  # Import class BookItem từ file BookItems.py

# Constants
MIN_PRICE = 0.5 # Định nghĩa giá trị nhỏ nhất cho giá sách hợp lệ
MAX_PRICE = 999.49 # Định nghĩa giá trị lớn nhất cho giá sách hợp lệ
CHUNK_SIZE = 1000 # Kích thước của mỗi chunk (khối dữ liệu) để xử lý

def from_datapoint(dp):
    """Xử lý một điểm dữ liệu (datapoint) riêng lẻ để tạo một đối tượng BookItem."""
    try:
        price = float(dp["price"]) # Chuyển đổi giá từ chuỗi sang số thực
        if MIN_PRICE <= price <= MAX_PRICE: # Kiểm tra xem giá có nằm trong phạm vi hợp lệ không
            book_item = BookItem(dp, price) # Tạo một đối tượng BookItem
            return book_item if book_item.include else None # Trả về đối tượng nếu nó hợp lệ, ngược lại trả về None
    except (ValueError, TypeError): # Bắt các lỗi nếu giá không thể chuyển đổi hoặc không có
        return None # Trả về None nếu có lỗi

def from_chunk(chunk):
    """Xử lý một chunk dữ liệu (một danh sách các datapoint)."""
    # Sử dụng list comprehension để xử lý từng datapoint trong chunk và lọc bỏ các giá trị None
    return [book for book in (from_datapoint(dp) for dp in chunk) if book]

def chunk_generator(rawdata, chunk_size=CHUNK_SIZE):
    """Một generator để chia dữ liệu thô (rawdata) thành các chunk có kích thước xác định."""
    size = len(rawdata) # Lấy tổng số lượng dữ liệu
    for i in range(0, size, chunk_size): # Lặp qua dữ liệu với bước nhảy bằng kích thước chunk
        # Sử dụng select để lấy một lát (slice) của dữ liệu, tạo thành một chunk
        yield rawdata.select(range(i, min(i + chunk_size, size)))

def load_books_from_rawdata(rawdata, workers=8):
    """Hàm chính để tải sách từ dữ liệu thô bằng cách xử lý song song."""
    results = [] # Khởi tạo danh sách để lưu trữ kết quả
    # Sử dụng ProcessPoolExecutor để tạo một pool gồm các tiến trình (workers)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        # Sử dụng pool.map để áp dụng hàm from_chunk cho từng chunk dữ liệu một cách song song
        # tqdm bọc ngoài để hiển thị thanh tiến trình
        for batch in tqdm(pool.map(from_chunk, chunk_generator(rawdata)), total=(len(rawdata) // CHUNK_SIZE) + 1):
            results.extend(batch) # Nối các kết quả từ mỗi batch vào danh sách chính
    return results # Trả về danh sách tất cả các đối tượng BookItem đã được xử lý