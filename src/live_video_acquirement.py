from aenum import Enum, extend_enum
import cv2
from datetime import datetime

class StreamSource(Enum):
    #BRAZILLIAN_BEACH = 'http://191.241.235.43/mjpg/video.mjpg' # might not be operating anymore
    DUTCH_STREET = 'http://62.131.207.209:8080/cam_1.cgi'
    AMERICAN_CROSSROAD = 'http://166.247.77.253:81/mjpg/video.mjpg'
    STRBA_SPORTS_PARK = 'http://stream.strba.sk:1935/strba/STRB_AREAL.stream/playlist.m3u8'
    GERMAN_MARKETPLACE = 'http://webcam.ehingen.de/cgi-bin/faststream.jpg?stream=full&fps=0'
    AMERICAN_AIRFIELD = 'http://flightcam1.pr.erau.edu/mjpg/video.mjpg'


class LiveVideo:
    def connect(self, destination: StreamSource):
        """
        Creates connection to live stream, enabling to capture its frames.

        Args:
            destination (StreamSource): input stream source
        """
        return cv2.VideoCapture(destination.value)

    def record(self, destination: StreamSource, fps: int = None, savename: str = None):
        """
        Enables recording and saving video from live stream. Follow the commands
        in displayed window title.

        Args:
            destination (StreamSource): input stream source
            fps (int): fps under which to save video (default are stream native
                        fps, might not correspond with number of frames actually
                        captured)
            savename (str): file name under which to save video (default is
                            derived from destination and current time)
        """
        stream = self.connect(destination)
        if fps is None:
            fps = stream.get(cv2.CAP_PROP_FPS)
        period = int(1000 / fps)

        if savename is None:
            current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
            destination_strings = str(destination).split(".")
            destination_name = destination_strings[1].lower()
            savename = f'{destination_name}_{current_time}'
        format_identifier = "mp4"
        file = f'{savename}.{format_identifier}'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(file, fourcc, stream.get(cv2.CAP_PROP_FPS), (int(stream.get(3)), int(stream.get(4))))
        saving = False

        while True:
            hasFrame, frame = stream.read()
            if not hasFrame:
                print("Live stream from this destination does not seem available at the moment")
                break
            cv2.imshow("Press 'r' to start recording, 's' to save recording and exit, 'ESC' to exit", frame)
            key = cv2.waitKey(period)
            if key == 27:
                break
            elif key == ord('r'):
                saving = True
            elif key == ord('s'):
                saving = False
                break
            if saving:
                video.write(frame)

        if int(video.get(cv2.CAP_PROP_FRAME_COUNT)) > 0:
            video.release()
        cv2.destroyAllWindows()

    def add_camera(self, name: str, address: str):
        """
        Adds stream source to StreamSource enumeration.

        Args:
            name (str): name under which to store the stream source
            address (str): address of connection to stream source
        """
        extend_enum(StreamSource, name, address)

if __name__ == '__main__':
    pass

    # EXAMPLE USAGE:
##    stream = LiveVideo()
##    stream.record(StreamSource.STRBA_SPORTS_PARK)