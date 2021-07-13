from dataloader.run_ocr import GoogleCloudOCR
from config.config_ocr import ConfigOCR


if __name__ == '__main__':
    config = ConfigOCR().get_config()
    ocr = GoogleCloudOCR(config)

    ocr.run_ocr_on_dataset()
