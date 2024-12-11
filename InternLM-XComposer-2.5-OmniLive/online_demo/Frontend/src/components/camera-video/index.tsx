import { useContext, useEffect, useRef, useState } from 'react';
import styles from './index.module.less';
import CameraIcon from '@/assets/svg/camera.svg';
import { ChatContext, IChatStatus } from '@/context/chat-context';
import { SOCKET_MESSAGE_MAP } from '@/config/message';
import useSrs from '@/hooks/useSrs';
import usePlayer from '@/hooks/usePlayer';
import { isMobile } from '@/utils/utils';
import classNames from 'classnames';
import { CHAT_SOCKET_URL } from '@/config/service-url';

export enum CameraStatus {
    DEFAULT = 1,
    ERROR = 2,
    AVAILABLE = 3,
}

let animationTimer: any = null;

const CameraVideo = () => {
    const [showMask, setShowMask] = useState<boolean>(false);
    const [showScanner, setShowScanner] = useState<boolean>(false);
    const socketRef = useRef<any>();
    const { sessionId, status, setStatus } = useContext(ChatContext);
    const { ref, srsReadyRef, cameraStatus, initSrs } = useSrs();
    const { playerRef, startFlag, closeAudio, readBuffer, createPlayer } = usePlayer();

    const handleSocket = async () => {
        if (socketRef.current) return;
        try {
            console.log('[camera-video] start to open socket');
            const socketUrl = CHAT_SOCKET_URL;
            socketRef.current = new WebSocket(socketUrl);
            socketRef.current.onopen = () => {
                console.log('[camera-video] socket onopen');
                socketRef.current.send(JSON.stringify({ session_id: sessionId }));
            };
            socketRef.current.onmessage = async (data: any) => {
                console.log('[camera-video] socket onmessage', data.data, socketRef.current);
                if (!data?.data) return;
                switch (data.data) {
                    case SOCKET_MESSAGE_MAP.FIRST_MESSAGE:
                        setStatus(IChatStatus.LISTENING);
                        break;
                    case SOCKET_MESSAGE_MAP.VOICE_START:
                        if (startFlag.current) return;
                        setStatus(IChatStatus.OUTPUTING);
                        if (!playerRef.current) {
                            createPlayer();
                        }
                        startFlag.current = true;
                        break;
                    case SOCKET_MESSAGE_MAP.INTERRUPT_VOICE:
                        console.info('[camera-video] close audio - force interrupted');
                        closeAudio();
                        break;
                    case SOCKET_MESSAGE_MAP.VOICE_END:
                        startFlag.current = false;
                        break;
                    case SOCKET_MESSAGE_MAP.SOCKET_FORCE_CLOSED:
                        console.info('[camera-video] close audio - socket closed');
                        closeAudio();
                        handleSocketError(IChatStatus.FORCE_CLOSE);
                        break;
                    default:
                        if (typeof data.data !== 'string' && startFlag) {
                            readBuffer(data.data);
                        }
                        break;
                }
            };
            socketRef.current.onerror = () => {
                handleSocketError(IChatStatus.SOCKET_ERROR);
            };
        } catch (e) {
            console.error('[camera-video] socket error', e);
        }
    };

    const handleSocketError = (status: IChatStatus) => {
        setStatus(status);
        socketRef.current && socketRef.current.close();
        socketRef.current = null;
        handleSocket();
    };

    const handlePrepareAnimation = () => {
        if (
            cameraStatus == CameraStatus.AVAILABLE &&
            (status === IChatStatus.PREPARING || status === IChatStatus.SOCKET_ERROR)
        ) {
            setShowMask(true);
            animationTimer = setTimeout(() => {
                setShowMask(false);
                setShowScanner(true);
            }, 3000);
        }
        if (status != IChatStatus.PREPARING && status != IChatStatus.SOCKET_ERROR) {
            if (animationTimer) {
                clearTimeout(animationTimer);
                animationTimer = null;
            }
            setShowMask(false);
            setShowScanner(false);
        }
    };

    useEffect(() => {
        if (sessionId && !srsReadyRef.current) {
            initSrs(handleSocket);
            srsReadyRef.current = true;
        }

        return () => {
            socketRef.current && socketRef.current.close();
            socketRef.current = null;
        };
    }, [sessionId]);

    useEffect(() => {
        handlePrepareAnimation();
    }, [cameraStatus, status]);

    return (
        <div className={classNames(styles.container, !isMobile() && styles.pc)}>
            {cameraStatus === CameraStatus.ERROR && (
                <div className={styles.containerError}>
                    <img src={CameraIcon} alt="" />
                    <div className={styles.containerErrorText}>{`开启摄像头和麦克风权限\n和我交流吧`}</div>
                </div>
            )}
            <div className={styles.containerVideo}>
                <video className={styles.video} playsInline autoPlay muted ref={ref} />
                {showMask && <div className={styles.mask}></div>}
                {showScanner && (
                    <div className={styles.scanner}>
                        <div className={styles.scannerInner}></div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default CameraVideo;
