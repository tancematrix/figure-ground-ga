"""
文字（列）を受け取って文字画像に変換する。
"""
import PIL.Image
import PIL.ImageFont
import PIL.ImageDraw

if __name__ == "__main__":
    fontsize = 150
    ttfontname = '/System/Library/Fonts/SFCompactDisplay-Light.otf'
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
    img.save("letter.jpg")