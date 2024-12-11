import { useContext, useMemo } from 'react';
import style from './top.module.less';
import { ChatContext, IChatStatus } from '@/context/chat-context';
import Prepare from '../prepare/prepare';
import AudioLoading from '../audio-loading/audio-loading';

const Top = () => {
    const { status } = useContext(ChatContext);

    const showPrepare = useMemo(() => {
        const prepareStatuses = [
            IChatStatus.PREPARING,
            IChatStatus.LISTENING,
            IChatStatus.FORCE_CLOSE,
            IChatStatus.SOCKET_ERROR,
        ];
        return prepareStatuses.includes(status);
    }, [status]);

    const showAudioLoading = useMemo(() => {
        return status === IChatStatus.OUTPUTING;
    }, [status]);

    return (
        <div className={style.top}>
            {showPrepare && <Prepare />}
            {showAudioLoading && <AudioLoading height={24} gap={4} forbidAnimate={false} />}
        </div>
    );
};

export default Top;
