import style from './home.module.less';
import CameraVideo from '@/components/camera-video';
import ChatProvider from '@/context/chat-context';
import Footer from '@/components/footer/footer';
import Top from '@/components/top/top';

const Home = () => {
    return (
        <ChatProvider>
            <div className={style.home}>
                <Top />
                <CameraVideo />
                <Footer />
            </div>
        </ChatProvider>
    );
};

export default Home;
