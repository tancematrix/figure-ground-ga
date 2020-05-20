# coding:utf-8
"""
文字（列）を受け取って文字画像に変換する。
"""
import PIL.Image
import PIL.ImageFont
import PIL.ImageDraw
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--font_path", action="store", type=str, default='/System/Library/Fonts/SFCompactDisplay-Light.otf', help="Path to your font. Both of ttf and otf work.")
    parser.add_argument("--output_file", "-o", action="store", type=str, default="letters.png", help="File name of the output image.")    
    args = parser.parse_args()

    fontsize = 150
    ttfontname = args.font_path
    text = input("文字を入力してください>> ")

    canvasSize = (100,100) # 仮に決めておく
    backgroundRGB = (255)
    textRGB       = (0)

    font = PIL.ImageFont.truetype(ttfontname, fontsize)

    # 以下、一回canvasに描画してみてtextの大きさを取得したのち、canvasSizeを更新してimg/drawを作り直している
    img  = PIL.Image.new('L', canvasSize, backgroundRGB)
    draw = PIL.ImageDraw.Draw(img)
    textWidth, textHeight = draw.textsize(text,font=font)
    canvasSize = (int(textWidth * 1.1), int(textHeight * 1.1))

    img  = PIL.Image.new('L', canvasSize, backgroundRGB)
    draw = PIL.ImageDraw.Draw(img)
    textTopLeft = (0 , -10)
    draw.text(textTopLeft, text, fill=textRGB, font=font)
    img_arr = np.array(img)

    # 文字のギリギリまでトリミング
    img_arr = img_arr[np.any(img_arr==0, axis=1)][:, np.any(img_arr==0, axis=0)]
    img = PIL.Image.fromarray(img_arr)
    img.save(args.output_file)