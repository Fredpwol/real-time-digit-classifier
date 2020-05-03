import tkinter as tk
from keras.models import load_model
from PIL import ImageGrab, Image
import numpy as np
import cv2
import os


model = load_model('mnist.h5')

class App(tk.Tk):
    def __init__(self):
        super().__init__()


        self.x = 0
        self.y = 0
        self.geometry('700x500')

        self.canvas = tk.Canvas(self, width=300, height=300, bg='white', cursor='cross')
        self.sidebar = tk.Frame(self)
        self.label = tk.Label(self.sidebar, text='input a number from 0 - 9.')
        self.guess = tk.Label(self.sidebar, text='')
        self.clear_button = tk.Button(self, text='clear', command=self.clear_board)
        self.sidebar.place(relheight=0.8, relwidth=0.4, relx=0.6)
        self.label.place(relx=0.3, rely=0.5)
        self.canvas.place(relheight=0.8, relwidth=0.6)
        self.clear_button.place(rely=0.85, relx=0.5)
        self.guess.place(relx=0.3, rely=0.2)

        # self.canvas.grid(row=0, column=0, pady=2, sticky='W')
        # self.label.grid(row=0, column=1, padx=2, pady=2)
        # self.clear_button.grid(row=1, column=0, padx=4 )
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.recognize)



    def clear_board(self):
        self.canvas.delete('all')
        self.guess.configure(text='')
        self.label.configure(text='input a number from 0 - 9.')

    def recognize(self, event):
        x = self.winfo_rootx() + self.canvas.winfo_x()
        y = self.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        
        ImageGrab.grab((x, y , x1, y1)).save('test.jpg')

        self.predict()


    def draw(self, event):
        self.x = event.x
        self.y = event.y
        radius = 8

        self.canvas.create_oval(self.x - radius, self.y - radius,
                                self.x + radius, self.y + radius, fill='black')
        
    
    def process_image(self):
        image = cv2.imread('test.jpg')
        grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 3) 
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)        
            # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
            cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)
            # Cropping out the digit from the image corresponding to the current contours in the for loop
            digit = thresh[y:y+h, x:x+w]        
            # Resizing that digit to (18, 18)
            resized_digit = cv2.resize(digit, (18,18))        
            # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
            padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)        
            # Adding the preprocessed digit to the list of preprocessed digits
            img = (padded_digit)

        return img
        

    def predict(self):
        img = self.process_image()

        img = img / 255.0

        img = img.reshape((1, 28, 28, 1))
        value = model.predict([img])[0], reverse=True
        
        os.remove('test.jpg')
        res = ''
        predicted = np.argmax(value)
        prob_ = max(value)*100
        self.guess.configure(text = f'{predicted}   {round(prob_, 2)}%', font=('Consolas', 24))
        scores = {}
        for num, prob in enumerate(value):
            scores[num] = prob

        for num, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
            res += f'{num} \t {round(prob*100,2)}%'
            res += '\n'
        self.label.configure(text=res)
        


if __name__ == "__main__":
    app = App()
    app.mainloop()