# -*- coding: utf-8 -*-
import wda
import time
import logging
import cv2
import numpy as np
import signal
import sys

CLICK_TARGET_TEMPLATE = [
    cv2.cvtColor(cv2.imread(f"resources/pi/{x}.png"), cv2.COLOR_BGR2GRAY)
    for x in ("red-bubble", "yellow-bubble")
]
THRESHOLD = 0.7


def detect_target(img_gray):
    result = []
    for template in CLICK_TARGET_TEMPLATE:
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= THRESHOLD)
        x = loc[0]
        y = loc[1]
        mask = np.zeros(img_gray.shape[:2], np.uint8)
        if len(x) and len(y):
            for pt in zip(*loc[::-1]):
                if mask[pt[1] + int(round(h / 2)), pt[0] + int(round(w / 2))] != 255:
                    x, y = pt[0], pt[1]
                    mask[pt[1] : pt[1] + h, pt[0] : pt[0] + w] = 255
                    logger.info(f"Click target detected at ({x}, {y})")
                    result.append((x, y, w, h))
    return result


if __name__ == "__main__":
    global c, logger
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s %(levelname)-8s] %(message)s"
    )
    wda.DEBUG = False
    logger = logging.getLogger(__name__)
    c = wda.USBClient()
    scale = c.scale
    while True:
        time.sleep(1.0)
        window_size = c.window_size()
        ws_w, ws_h = window_size.width, window_size.height
        result = detect_target(
            cv2.cvtColor(np.asarray(c.screenshot()), cv2.COLOR_RGB2GRAY)
        )
        if len(result):
            logger.info(f"{len(result)} targets detected")
        for i, (x, y, w, h) in enumerate(result):
            tx, ty = int((x + w // 3) // scale), int((y + h // 3) // scale)
            start = time.time()
            c.tap(tx, ty)
            logger.info(
                f"target {i}: tapping at ({tx}, {ty}), time cost: {time.time() - start}"
            )
    c.close()
