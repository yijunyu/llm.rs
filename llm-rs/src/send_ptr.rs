#[derive(Debug, Clone, Copy)]
pub struct SendPtr<T> { pub ptr: *mut T }

unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    pub fn new(ptr: *mut T) -> SendPtr<T> {
        SendPtr { ptr }
    }
}