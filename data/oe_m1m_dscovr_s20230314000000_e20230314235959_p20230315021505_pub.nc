CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230314000000_e20230314235959_p20230315021505_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-03-15T02:15:05.614Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-03-14T00:00:00.000Z   time_coverage_end         2023-03-14T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxm֪�  
�          @����@A��ȣ��m�HB�����@���k����HB�p�                                    Bxmֹ&  �          @�\��@3�
��z��qffB�����@����l(���B�W
                                    Bxm���  
Z          @���Q�@j=q�Å�Y��B�
=�Q�@�33�Mp���Q�B�                                    Bxm��r  "          @��z�@�33��
=�E
=B�B��z�@�\)�!G����B��                                    Bxm��  �          @�p���
@�=q���
�9z�B���
@�\)�ff����B��f                                    Bxm��  �          @�\�33@~{����R=qB�#��33@љ��<(����Bڣ�                                    Bxm�d  �          @�=q�G�@Vff���k33B�aH�G�@����w
=��33B�#�                                    Bxm�
  �          @�G��z�@.�R�ָR�{B�z��z�@�z���
=�33B���                                    Bxm��  �          @��H��\)@=q����B�����\)@�33�����B�                                      Bxm�.V  T          @�R��Q�@Q����
��B��
��Q�@�p���G����B�33                                    Bxm�<�  T          @�  �
=q@A�����up�B����
=q@�{��z��=qB�p�                                    Bxm�K�  
�          @�
=���@Mp��Ӆ�kG�B�aH���@�\)�xQ���p�B��                                    Bxm�ZH  
Z          @���Q�@7
=��ff�s��C�Q�@����z����B�W
                                    Bxm�h�  
�          @����z�@0  ��p��z33C!H�z�@�Q���z��	(�B�#�                                    Bxm�w�  "          @���z�@'
=����z�C c��z�@�(���ff�{BڸR                                    Bxm׆:  �          @�
=���@Z=q��=q�i{B�Ǯ���@�(��o\)��{B��                                    Bxmה�  �          @����@=p������p�HB�L���@�\)�}p����B���                                    Bxmף�  
�          @��ff@.{��(��v(�C�R�ff@��H��p���B���                                    Bxmײ,  �          @��
=@0  ��{�v�C���
=@�z���ff�
=Bߨ�                                    Bxm���  �          @����\@'��ָR�zffC(���\@�����G��
p�B�.                                    Bxm��x  T          @���Q�@
=��G�.C&f�Q�@�
=��G����Bޔ{                                    Bxm��  �          @��\)@
=�љ��C���\)@�Q������B�{                                    Bxm���  �          @���Q�@$z�����}Q�B���Q�@��\����
�B�=q                                    Bxm��j  �          @��ٙ�@.�R����}ffB�Ǯ�ٙ�@��R�~�R���B��                                    Bxm�
  �          @�=q��=q@   ���H��HB�uÿ�=q@�\)�����{B��                                    Bxm��  T          @�(���p�@{��(��~�C �R��p�@�\)��33���B۽q                                    Bxm�'\  "          @ᙚ���
@{��p�\)B�z���
@����(����B�
=                                    Bxm�6  T          @�33�
=@,���ƸR�t=qC   �
=@��\�u��B�u�                                    Bxm�D�  
          @�33�\)@$z���
=�u{C&f�\)@�\)�z�H�  B��{                                    Bxm�SN  
�          @�ff��@&ff���
�x�HC@ ��@�=q�����
=Bݮ                                    Bxm�a�  T          @�Q��   @'��θR�{�
B��
�   @�z���33�	�B��                                    Bxm�p�  T          @����(�@����p���C)�(�@����  ���B���                                    Bxm�@  
�          @�����?����33\)C�\���@��H��33��B�p�                                    Bxm؍�  
Z          @�\�p�?�������
=C�\�p�@�����=q�{B�                                    Bxm؜�  T          @��.{?�=q��R\C#��.{@������"ffB��H                                    Bxmث2  
�          @�
=�<(�?������Q�C���<(�@�������\B�\                                    Bxmع�  	`          @�{�;�?������
�~�CT{�;�@�p��������B�                                    Bxm��~  �          A{�E�?��
���
=C���E�@��
�����!�B�R                                    Bxm��$  
�          A ���N�R?�(�����{�C���N�R@����� =qB�\)                                    Bxm���  �          @���Mp�?�Q����{��C8R�Mp�@�{����� ��B�=                                    Bxm��p  �          A  �X��?����z��\)C���X��@�  ���H�+{B�(�                                    Bxm�  
�          A	��qG�?�����
�{{C#�H�qG�@�{�����.(�B���                                    Bxm��  �          A	�qG�?�(���z��zG�C"
=�qG�@����33�+G�B�aH                                    Bxm� b  T          A33�c33?�����\k�C���c33@����
=�-Q�B�=q                                    Bxm�/  T          A
=q�\(�?��R����� C���\(�@�����\�)�B�u�                                    Bxm�=�  �          A	��Y��?�Q���G�W
C��Y��@�=q��(��+��B�#�                                    Bxm�LT  T          @���G
=?�(����
��C�=�G
=@�����p��.33B�                                      Bxm�Z�  "          A=q�H��?����{  C!���H��@�G������5  B��=                                    Bxm�i�  �          A	���`��?:�H����C(O\�`��@�
=�ȣ��;=qB�L�                                    Bxm�xF  
�          A��dz�?.{� z�G�C).�dz�@�����\)�=(�B�\)                                    Bxmن�  
(          A�h��?
=q���fC+�\�h��@�{�ҏ\�?B��                                    Bxmٕ�  "          A���\(�?��{��C*���\(�@�  ��(��A�B��q                                    Bxm٤8  �          Ap��n�R>�ff�   ��C-
�n�R@��\�ҏ\�@Q�C0�                                    Bxmٲ�  "          A�qG�>���� (�k�C/  �qG�@�\)�����B��C)                                    Bxm���  T          A{�j�H>\���)C.��j�H@������C=qC ��                                    Bxm��*  "          A\)�xQ�?������|C+�f�xQ�@�����=q�:Q�C}q                                    Bxm���  "          A  �\)?.{��  �yC*ff�\)@�(���Q��6�C                                    Bxm��v  
�          A	�l��?�R��  W
C*u��l��@��\�����;p�C �                                    Bxm��  
�          A
ff�`  ?5���
.C(���`  @�
=���H�<B�#�                                    Bxm�
�  "          A��hQ�>k����3C0s3�hQ�@��\�����Fz�C��                                    Bxm�h  "          A�
�e�>8Q����R� C1{�e�@����ff�HQ�C�q                                    Bxm�(  �          A�\�u�?�R� z�k�C*�u�@�
=�����<33C\                                    Bxm�6�  T          A
=�e�?\)����qC+��e�@�ff���
�A  B���                                    Bxm�EZ  "          A{�n{@�z�����gG�B��n{@��
��  ��=qB��                                    Bxm�T   
�          A���W
=@�\)���dz�B�{�W
=@���xQ���=qB�{                                    Bxm�b�  T          A  �L��@�G���{�h�HB���L��@�  �\)���B��
                                    Bxm�qL  T          A��^�R@������m�Bʀ �^�R@�(������\)B���                                    Bxm��  
�          A\)�W
=@����33�s\)B��H�W
=@�  ���\���HB���                                    Bxmڎ�  �          A�\(�@���Q��t�
B���\(�@�33���R��{B��
                                    Bxmڝ>            Aff�B�\@��������s(�B���B�\@�{��p���(�B�G�                                    Bxmګ�  	�          A\)��=q@s33� z��{Bӳ3��=q@�p���z��p�BĔ{                                    Bxmں�  "          A(��h��@tz����G�B���h��@�
=��ff�\)B��f                                    Bxm��0  "          A���z�@z=q����}(�B��῔z�@�G�������B�p�                                    Bxm���  T          A�ÿ��@o\)�ff{BڸR���@�p�������B��                                    Bxm��|  �          A��@Vff����fB�녿�@�{����B�W
                                    Bxm��"  "          A�ÿ���@=p��33�)B�(�����@�����ff�{B�33                                    Bxm��  T          Aff��Q�@=p�����B߮��Q�@޸R�������B�\)                                    Bxm�n  �          A\)��G�@'
=��
G�B���G�@�  ��z��'(�B�8R                                    Bxm�!  
�          A�R��\)@�R�(��B�=��\)@�z���
=�*�B��                                    Bxm�/�  "          AQ�s33@p��=qB�LͿs33@�{��33�,�HB�(�                                    Bxm�>`  
�          A����\@�
�  �)B�8R���\@�������6  B�k�                                    Bxm�M  
�          A�\��ff@G�����B�Q쿦ff@Ӆ��=q�1G�B�33                                    Bxm�[�  T          A�\���
@!G���
=B�\���
@��������+ffB�u�                                    Bxm�jR  
�          A���ff@  ��B��ÿ�ff@��
����2ffB�8R                                    Bxm�x�  
�          A  ��z�@ ����B�33��z�@θR��(��9G�B�k�                                    Bxmۇ�  
�          AG��\?��H���k�C�H�\@Ǯ��33�?z�B�Q�                                    BxmۖD  "          A33�˅?�
=��(�Ch��˅@�
=�ᙚ�:�\BϏ\                                    Bxmۤ�  �          A����@	���=q�RB�𤿫�@�p���ff�6�\B���                                    Bxm۳�  �          A�\����?�p��p�B�������@Ϯ��Q��:33B���                                    Bxm��6  
�          A����R@�
��RW
B�Q쿞�R@��H�����9ffB�aH                                    Bxm���  T          A�\���@
=�Q�ffB�W
���@أ���Q��1�\B�k�                                    Bxm�߂  
�          A���G�?�
=��Ru�B�G���G�@�\)���<ffB�
=                                    Bxm��(  �          A녿�z�?��
=CT{��z�@��\��Q��M�\B�8R                                    Bxm���  
�          A��n{��=q�(�¨�CD+��n{@�  ����q�RB�=q                                    Bxm�t            AQ��(�@,����z��3B�33��(�@�
=��{��B�#�                                    Bxm�  
�          Aff��  @K���=q�~z�B����  @�
=��p���\B�G�                                    Bxm�(�  �          A
ff��@g
=���zp�B�R��@�������B���                                    Bxm�7f  "          A���\)@e���R�{�HB���\)@��������B���                                    Bxm�F  T          AQ��{@w���\)�y=qB����{@�z������B̅                                    Bxm�T�  
�          A�׿�@x������x=qB�uÿ�@�������(�B�ff                                    Bxm�cX  
�          Az�ٙ�@���=q�o��B�LͿٙ�@�\�����p�B�W
                                    Bxm�q�  "          A녿�  @�������n33B�uÿ�  @�p���33��Q�Bͨ�                                    Bxm܀�  �          A��
=@�Q���\)�h\)B����
=@�=q���
��p�B�B�                                    Bxm܏J  
�          A�\��(�@�Q���z��a�Bۣ׿�(�@�\)��p��؏\B�G�                                    Bxmܝ�  "          AQ���H@������^��B�33���HA{��33��  B˸R                                    Bxmܬ�  �          A���  @������a��B����  A (���ff����B̔{                                    Bxmܻ<  "          A�\�ٙ�@����  �g\)B�녿ٙ�@�����
��ffB�u�                                    Bxm���  �          A��(�@��H��  �iB��)��(�@����  ����B��
                                    Bxm�؈  "          A�R����@�z����c�
Bߙ�����@�(��������B�z�                                    Bxm��.  T          Aff��@�����ff�e=qB�z��@���������=qB��
                                    Bxm���  �          A
=���
@��
��G��h{B؏\���
@�p��������HB���                                    Bxm�z  �          A33��z�@��\��ff�cp�Bԣ׿�z�A ����
=��z�B��f                                    Bxm�   
�          A\)��ff@�����H�]��B�ff��ffA{��G���=qBɅ                                    Bxm�!�  T          Az��\@�G�����b=qBܣ׿�\A �������ۮB��f                                    Bxm�0l  "          Az���@�{���
�]  B�  ���A�������=qB�W
                                    Bxm�?  �          A{��@�\)��R�[z�B����A ���|����G�B�\)                                    Bxm�M�  �          AG���(�@�33���
�X�
B�(���(�A�s�
��z�Bȏ\                                    Bxm�\^  �          A�Ϳ�=q@������V�HB�k���=qA��u��33BɊ=                                    Bxm�k  	�          A���p�@������b��B����p�@�(���=q���B�#�                                    Bxm�y�  
�          A���
@�(���
=�^=qB��H��
Ap�����ׅBУ�                                    Bxm݈P  T          A�
��z�@�G��ᙚ�M��B۸R��z�A�]p���Q�BΊ=                                    Bxmݖ�  
�          Az��{@���љ��R�B٨���{@�(��S�
��G�B̳3                                    Bxmݥ�  T          A����@�
=�   �f  B�G���Ap���=q��\BθR                                    BxmݴB  
�          Aff���
@�ff��
=�r��B�\���
@��
��=q��ffBʸR                                    Bxm���  �          Ap����
@��H��ff�PB�p����
@�(��_\)����B�ff                                    Bxm�ю  
�          A	G���=q@�(����
�J=qB�G���=q@�  �L(���z�BΙ�                                    Bxm��4  �          A  ��=q@����أ��S\)Bݳ3��=q@����_\)���
B�G�                                    Bxm���  	�          A33��z�@����˅�K�B�z��z�@�\�HQ����
B�G�                                    Bxm���  
�          A=q��33@�G�����)p�Bճ3��33A�H���X  B�p�                                    Bxm�&  T          A33� ��@�z����
� {B��f� ��Ap����H�4z�B�{                                    Bxm��  
N          @�
=���@��
���H�F(�B�L����@�{�9�����
B�G�                                    Bxm�)r  
          A�H�ff@�z������-�B�B��ff@����\)�v�RBծ                                    Bxm�8  
�          A��Q�@�p���=q�$�\B��
�Q�@��Ϳ�z��P��BՊ=                                    Bxm�F�  	�          A=q�{@���ff�
�\B�ff�{A  ��  �ə�Bә�                                    Bxm�Ud  
          A{�
�H@�33���R�B�33�
�HAzῑ����BД{                                    Bxm�d
  
�          A�ff@Ϯ�����B��
�ffA �Ϳ��
���B�=q                                    Bxm�r�  
N          Az���@�
=��G���ffB���@��R��
=�;�BҮ                                    BxmށV  
          A
�R�/\)@�G��tz�����B�8R�/\)A�>�=q?��B��H                                    Bxmޏ�  "          Aff���@�  �[���33B�L����A   ?z�@~�RB�                                      Bxmޞ�  �          A	p��#33@���J�H���B�L��#33A=q?u@�{B���                                    BxmޭH  T          A���9��@�{��
�{\)B�\)�9��@��\?�G�A>�HBܨ�                                    Bxm޻�  
�          A{�A�@��Ϳ��R���B��\�A�@ᙚ@#33A�33B�                                    Bxm�ʔ  
�          A��HQ�@�(���ff��ffB�  �HQ�@�ff@-p�A�ffB�\                                    Bxm��:  �          @�  �333@���G���33B�#��333@�(�?�z�A&�HB��f                                    Bxm���  "          Az��B�\@��
�����{B����B�\A�
���QG�B�L�                                    Bxm���  T          A��8��@�Q���G��뙚B�#��8��AQ쾔z��B؀                                     Bxm�,  �          A�H��@�\�X����\)B�����@��\?   @`��B՞�                                    Bxm��  "          A(��$z�@����:=q����B�B��$z�@�(�?���@�(�B�{                                    Bxm�"x  
Z          A�\�5�@�Q��U��  B���5�Aff?^�R@�33B�G�                                    Bxm�1  "          A�\�#�
@��H�g
=����B۞��#�
@��R>���@�
Bי�                                    Bxm�?�  
�          A
=�,(�@�Q��|����Q�Bܞ��,(�A(�=��
?�B�                                      Bxm�Nj  "          A�\���@�ff����G�B�8R���A��:�H��
=B�=q                                    Bxm�]  
�          A���3�
@�(���  ��  B�u��3�
A���p��   B�aH                                    Bxm�k�  
�          A�1�@��
�Z=q���B�{�1�@��>�33@�B��f                                    Bxm�z\  �          A ���`  @ڏ\��
�p  B�G��`  @�
=?�G�A/\)B�W
                                    Bxm߉  
�          A
=q�u@��H@��A���B�33�u@���@���B#�RB�33                                    Bxmߗ�  �          A
{�a�@��
@1�A�=qB�8R�a�@��@�  B0  B�W
                                    BxmߦN  
�          A�����@�33@,��A�z�B������@�(�@��B(ffC Y�                                    Bxmߴ�  
�          A{���\@�p�@�G�AҸRB�����\@��@�(�BD(�C��                                    Bxm�Ú  �          Az��tz�@�@UA�(�B�z��tz�@���@�  B9=qB��                                    Bxm��@  �          A
=��{@��@@��A��HB�z���{@w
=@��B-�C                                    Bxm���  T          Ap�����@�R@p�A�B������@�33@�  B"��B���                                    Bxm��  T          A���k�@�G�?�Q�A0��B�\�k�@���@�
=B  B��)                                    Bxm��2  �          @��K�@�Q�@~�RB�
B�\)�K�@,��@���B]{C�3                                    Bxm��  "          @���(�@�{@�G�BB��B��)��(�?��R@߮B�#�C�\                                    Bxm�~  "          @�p��G�@�p�@��B:�B�p��G�?���@���B�{CB�                                    Bxm�*$  T          @�
=�#33@�@�\)B�\B�Q��#33@&ff@��HB{33C\)                                    Bxm�8�  T          A33�1�@�ff@R�\A�(�Bߞ��1�@���@ƸRBE�B��                                    Bxm�Gp  
�          A��fff@��@R�\A�{B�{�fff@�G�@�\)B933B�                                      Bxm�V  �          Az��k�@�@]p�A��B�=q�k�@��@θRB=��C G�                                    Bxm�d�  
�          A
=�k�@�p�@J=qA�33B�aH�k�@���@�{B6�B�                                    Bxm�sb  "          A=q����@�G�@s�
A�=qB�#�����@n{@�33BD��C	5�                                    Bxm��  
�          A33��=q@���@>{A���B����=q@���@��\B.�HC��                                    Bxm���  
N          A��{�@�ff@{Al  B���{�@�  @��B=qB��                                    Bxm��T  .          A�����\@��
?��A"=qB�Q����\@�
=@�
=B
{B���                                    Bxm��  
�          A���(�@�ff?\A=qB�ff��(�@��@��B��B��=                                    Bxm༠  
�          A�����@�{?���A$Q�B�B�����@���@���B
�B�u�                                    Bxm��F  �          Ap��z=q@���?�G�AG�B�p��z=q@�z�@�  B

=B�
=                                    Bxm���  �          Az��:�HA ��?���A@  B��H�:�H@Ǯ@��B�\B�33                                    Bxm��  �          Ap��k�@���?�(�AM�B�=�k�@�{@��B
=B�=                                    Bxm��8  T          A������@�  @A\��B�R����@�(�@�(�B=qB�p�                                    Bxm��  �          A���G�@�p�?��AG�B�����G�@�{@���B�C��                                    Bxm��  "          A\)���@�\?�
=@���B��f���@�z�@��A��B���                                    Bxm�#*  
�          A33���@�R?Q�@���B�����@�ff@�
=A���B�B�                                    Bxm�1�  z          A����\@��H?+�@�=qB������\@Ӆ@���A�=qB�W
                                    Bxm�@v  
�          A  �z�HAG�?��@u�B��z�H@�33@�=qA�G�B�                                    Bxm�O  �          A
=�y��A ��>k�?��HB��y��@�  @~{Aң�B�.                                    Bxm�]�  �          A�R�w
=A z�>��
@33B�aH�w
=@�ff@���A�33B�#�                                    Bxm�lh  H          Aff�p��A����Ϳ(��B�  �p��@�@j�HA�
=B�G�                                    Bxm�{  
�          A
=�l(�A=q�#�
��=qB��l(�@�
=@p��A���B�#�                                    Bxmቴ  T          A�R�qG�AG����
���HB���qG�@�@l(�A�B�p�                                    Bxm�Z  �          A��w
=A   =�\)>�
=B�=�w
=@ᙚ@q�A�p�B�k�                                    Bxm�   �          A��b�\A>�G�@6ffB�.�b�\@�\)@��A��
B��
                                    Bxmᵦ  "          Aff�x��A   ���Ϳ+�B��H�x��@�(�@g�A�{B�8R                                    Bxm��L  �          A{��=q@��
�\)�fffB����=q@�\)@I��A�z�B�3                                    Bxm���  �          A�R���\@����
=��B�����\@�ff@�Am�B�L�                                    Bxm��  "          A{����@����H��B������@�p�@�RAh��B�.                                    Bxm��>  T          Ap���\)@���  �Q�B����\)@�ff@(�Ae�B�33                                    Bxm���  T          A����R@��H���R�\)B����R@�p�@�AeB�#�                                    Bxm��  �          A  �|(�@���z��[�
B�Q��|(�@�(�?�\)A+
=B��                                    Bxm�0  �          A�
�x��@����G��33B��)�x��@�@�Ah(�B��H                                    Bxm�*�  "          A
=q��ff@�\)��G��׮B��ff@�@"�\A���B��                                    Bxm�9|  �          A	�����\@�G�����l(�B��)���\@�ff@>{A�B��
                                    Bxm�H"  
�          A�
�|(�@��
���R�!�B�=q�|(�@�\)@33A`  B�#�                                    Bxm�V�  T          A	p����@���a��B�#����@��H?�A��B�
=                                    Bxm�en  "          A
=q��33@�녿�
=�N�HB�(���33@��?�G�A ��B�ff                                    Bxm�t  T          A	�����@��Ϳ�Q��PQ�B�=q����@�?��A$z�B�\                                    Bxm₺  	�          A����=q@��H�G��[
=B�R��=q@�\)?�
=AffB�3                                    Bxm�`  "          A����@�p������z�B������@Ϯ?�(�A[�B�(�                                    Bxm�  �          A
=q��Q�@����(��k33B�aH��Q�@�Q�?�  A��B�q                                    Bxm⮬  	�          A
=���@������x��B������@�33?�  @�(�B�W
                                    Bxm�R  �          A�
��\)@޸R�ff�yG�B�k���\)@��?���@��B�#�                                    Bxm���  T          A�
��  @��H�L(�����B����  @�G�>��?�p�B�                                    Bxm�ڞ  �          A��qG�@��������G�B����qG�@��H�s33��  B�aH                                    Bxm��D  �          @���z�@�33@C33A�z�Bݏ\�z�@�=q@�\)BF{B�\)                                    Bxm���  "          @��
�9��@�Q�@Z�HA�{B�u��9��@W�@�G�BL�C�f                                    Bxm��  
�          @�33�1G�@��@QG�A�BꞸ�1G�@S�
@��\BL
=C��                                    Bxm�6  "          @ٙ��8Q�@�  @G�A�z�B�\�8Q�@Q�@���BF�CQ�                                    Bxm�#�  
�          @����-p�@���@�RA\B�=q�-p�@I��@��B9�\C                                    Bxm�2�  �          @�  �*�H@�33@�A�33B�W
�*�H@_\)@��B4\)B��f                                    Bxm�A(  
�          @�
=�-p�@�33�
=��\)B��-p�@�\)?�\@��HB��                                    Bxm�O�  T          @��0  @�
=��{��\)B�W
�0  @�\)?G�@�G�B�G�                                    Bxm�^t  �          @����%@����(������B�G��%@��
=�G�?uB��f                                    Bxm�m  �          @�Q��%�@����W���ffB�  �%�@�\)�&ff����B��                                    Bxm�{�  �          @�p����H@������{Bߙ����H@�ff��Q��H(�B��                                    Bxm�f  "          @�����@�����0BЮ����@�p��   ���RBɔ{                                    Bxm�  T          @�p��
=q@�p����H�*��B�33�
=q@��
�����B��                                    Bxm㧲  "          @�G����R@����ff�C
=B��׾��R@���������B�(�                                    Bxm�X  �          @�p��\@�ff��  �;��B�W
�\@�Q��
�H���B��=                                    Bxm���  
�          @θR�Tz�@����
�6B�LͿTz�@�Q�����B�                                    Bxm�Ӥ  
�          @�  ����@����  �2�HB�=q����@��������  B�aH                                    Bxm��J  
f          A z���@������H�&��B�R��@���
=���\B�L�                                    Bxm���  �          A33�Fff@�������G�B�W
�Fff@�����<z�B�ff                                    Bxm���  �          A��W�@�
=��p��z�B��W�@�{���>=qB�8R                                    Bxm�<  �          A
=�c33@��
��\)��RB�L��c33@���Q��"ffB�.                                    Bxm��  
�          A��z�H@��
���
��{B�G��z�H@�녿�G���p�B�p�                                    Bxm�+�  �          A(���(�@���y���ڏ\B��f��(�@��O\)��{B�W
                                    Bxm�:.  �          A	����\@�{�g���=qB��f���\@�(���G��;�B��H                                    Bxm�H�  �          A
�R��ff@����e����HB��=��ff@�{��Q��Q�B�Ǯ                                    Bxm�Wz  "          A0Q���33A	������B�=��33A  �����HB�=q                                    Bxm�f   "          AM���A ����33��z�B�B����A9���aG��z�HB��                                    Bxm�t�  �          AP(����
A#\)��  ��(�B��3���
A;33�:�H�N�RB�\)                                    Bxm�l  
�          AO
=��{A.=q�|�����\B�.��{A8Q�?�
=@��B�\                                    Bxm�            ATz����
A.=q������B� ���
AB�\����B�                                    Bxm䠸  |          AQ���ffA1����=q��33B�
=��ffAAp�>��@�
B��                                     Bxm�^  �          AQ����HA@���.�R�@��Bۨ����HA@z�@0  AB=qBۨ�                                    Bxm�  �          AO33��
=A<z��=p��S33Bݔ{��
=A>ff@�A-�B�33                                    Bxm�̪  �          AQG����A>�\��H�*ffB����A<z�@=p�AQB�p�                                    Bxm��P  
�          AU����AE���8Q��FffB�L�����A733@�A��\B�W
                                    Bxm���  �          AT  ��  A@��?Ǯ@�  B����  A#\)@��HA�
=B�u�                                    Bxm���  T          AU����ADQ�?�ff@�Q�B�����A)�@�
=A�{B�                                    Bxm�B  T          AT������AD  ?�=q@���B������A)��@�\)A��
B�                                    Bxm��  �          AU����AE�?�@���Bߔ{��A*{@��HA�G�B�R                                    Bxm�$�  �          AT�����RAAG�@	��A��B�����RA�
@�B{B�
=                                    Bxm�34  �          AQp���z�A<Q�@!�A2ffB�.��z�A��@�=qB  B�=                                    Bxm�A�  �          AS\)���HA?
=@33A ��B�33���HA��@�B��B��
                                    Bxm�P�  T          AK33��ffA8Q��{�3\)B�W
��ffA7�@+�AC
=B��                                     Bxm�_&  �          A?
=��z�A,�ÿ�  �33B�k���z�A(z�@AG�Ak\)B�=                                    Bxm�m�  @          A1��(�A �׿�  �ϮB����(�A�@H��A�Q�B��                                    Bxm�|r            AW���33@�A-G�BWffB�Ǯ��33@
=AQ�B��fC ��                                    Bxm�  T          AZ�H��  A�A,z�BP�BȨ���  @8��AT  B�aHB��)                                    Bxm噾  "          AS����HA�
A�B6BɅ���H@�AD��B���B�p�                                    Bxm�d  �          AP�Ϳ(�AffA#
=BM
=B�� �(�@EAJ�HB�#�B�=q                                    Bxm�
  T          Ac���(�A��A(��B>\)B=��(�@��RAYG�B�33B�B�                                    Bxm�Ű  �          Ai���
A2=qA�HB(�B����
@ǮAPQ�B|��B�z�                                    Bxm��V  "          Abff�Z�HA ��A  B!{Bٙ��Z�H@��HAFffB{��B�Q�                                    Bxm���  �          Ao\)��AG�@�  A��B�����A ��ADQ�B_
=B�=q                                    Bxm��  �          Aw33�p�A[
=@�(�A�ffB�W
�p�AffA<(�BG��B���                                    Bxm� H  
�          Aw��i��Ai�������z�B�\�i��A[\)@��
A��RB�Ǯ                                    Bxm��  T          A|(���=qAmp��33��RB�33��=qAg
=@��As
=B��                                    Bxm��  �          A~ff�AG�AhQ���Q����Bˀ �AG�Aw�
?�ff@s33B�
=                                    Bxm�,:  
�          A��
�p��AV�R�33���B�
=�p��A~�H�#33�G�B�Ǯ                                    Bxm�:�  �          A~{����AQ���p��\)B�������Ayp��%�{B���                                    Bxm�I�  �          A|���Mp�A_�
��  ����B��
�Mp�Av�R���;\B˅                                    Bxm�X,  "          Av=q�O\)Aa��������B��H�O\)Ao�?�33@�G�B�k�                                    Bxm�f�  
�          As�
�o\)AS33���H��33Bӣ��o\)Al  ��� ��B�z�                                    Bxm�ux  
          Au���j�HANff��
=����Bӽq�j�HAmG���(����B�Ǯ                                    Bxm�  
          Au��Q�ALz����H��(�B��
��Q�Ai���Q���
=BԔ{                                    Bxm��  �          Apz���{AP�����H���Bר���{Af=q�.{�(��BԀ                                     Bxm�j  
�          Au����RAX  ���R��p�B�aH���RAg33?L��@@  B��                                    Bxm�  �          At����p�AO�
��p���G�Bٔ{��p�AiG��=p��2�\BոR                                    Bxm澶  J          Af�R�i��AB�H��z���p�B�W
�i��A[��Tz��VffB�                                    Bxm��\  �          AN=q��z�A6{��z����B�q��z�A2�R@:�HAR{B鞸                                    Bxm��  �          AN�R��{A7�
��������B����{A/33@j�HA��\B��f                                    Bxm��  T          AK33��ffA3\)��G����B��f��ffA*ff@i��A�\)B�aH                                    Bxm��N  �          AG���=qA.=q�Vff�yG�B����=qA4��?���@��HB�L�                                    Bxm��  �          AQ���ۅA z���z���p�B��3�ۅA4  �(��*�HB�3                                    Bxm��  �          A[33��  A0Q���(���ffB�#���  AB�\�W
=�fffB�u�                                    Bxm�%@  "          Ac\)���HA/
=������B�����HAPz��=q�Q�Bޮ                                    Bxm�3�  T          Ab=q���\A*�H���
��B����\AO33�7��;�
B��                                    Bxm�B�  �          A^{��Q�A&{��z��
��B�=q��Q�AJ�H�AG��IG�B��                                    Bxm�Q2  "          A^=q����A$(���H��B㞸����AK
=�Vff�_\)B�B�                                    Bxm�_�  "          A]���tz�A!���
ff��RB�k��tz�AK��u��
=B�ff                                    Bxm�n~  "          A\���fffA*{� (��
=B�k��fffAO33�C�
�L��B�\                                    Bxm�}$  �          A^=q�~�RA&�\�����\B����~�RAM��Z=q�c\)B�aH                                    Bxm��  �          A`�����A$���	G���\B������AM�o\)�v�RB؀                                     Bxm�p  �          Ab=q��{A&ff���
=B�{��{AN{�`���e��B�
=                                    Bxm�  "          Ae���z�A)p��
=��B�\)��z�AQG��`���b�HB��                                    Bxm緼  �          AiG���G�A(���p��ffB�Ǯ��G�AS33�y���x(�B��
                                    Bxm��b  
�          An=q���
A)���{�(�B�\���
AW\)��(����HB��)                                    Bxm��  �          Aj=q��z�A*{����B��)��z�AUp���=q��G�B�ff                                    Bxm��  T          AhQ���=qA$z��{�!\)B�(���=qAR�\������B�W
                                    Bxm��T  "          Ai���Ap��"=q�/�
B�\)���AM���Q���33B�#�                                    Bxm� �  �          AmG��~�RA�
�&ff�2��B�z��~�RAP����ff��33B��                                    Bxm��  
�          Al  ����A���'
=�4Q�B�k�����AO
=������G�Bր                                     Bxm�F  "          AiG��w�A  �)p��:=qB�k��w�AJ�H�\��33B���                                    Bxm�,�  T          AeG��J=qA���0���I�B܊=�J=qAC��ٙ���ffB�                                      Bxm�;�  
�          AZ�H�Z=qA33��R�5�Bݳ3�Z=qA@z����
��{Bӣ�                                    Bxm�J8  
�          AE�J=qA:�R�!G��<��B�G��J=qA;�@��A(��B�#�                                    Bxm�X�  
�          AH(����A@�׿���(�B����A=�@>{A]p�B�p�                                    Bxm�g�  T          AL���!�A?
=�z=q��(�B����!�AH(�?���@���B��f                                    Bxm�v*  "          AMp��:�HA:ff������  B���:�HAH  >�Q�?���B�B�                                    Bxm��  �          AO\)�R�\A0Q�������G�B�33�R�\AH  ��G����B�u�                                    Bxm�v  T          AU���c�
A4(���ff�ݮB�{�c�
AL�׿�=q���B��                                    Bxm�  
�          A\Q��S33A3���  ��=qB�Ǯ�S33AR�R�z��\)B�(�                                    Bxm��  T          A[��EA"�H�33�(�B��EAL(����\��  B�8R                                    Bxm�h  T          AR=q�i��@��� ���Gp�B�  �i��A/33�����G�B��H                                    Bxm��  T          AR=q�g
=@�
=����A  B����g
=A2=q���R���B���                                    Bxm�ܴ  
�          ATz��P��A���z��>  Bޮ�P��A7�����υBӽq                                    Bxm��Z  
Z          AS33�FffA���G��3��B��
�FffA;\)��
=��z�BѨ�                                    Bxm��   �          AT���<��A�33�5p�B��
�<��A<����=q���HB�
=                                    Bxm��  �          AT���EAff�ff�-��B�L��EA?\)��{��=qB���                                    Bxm�L  �          AS�
�G�A\)�\)�$�B�Q��G�AD����=q��z�B��f                                    Bxm�%�  �          AU���333A
=�(��)��B��333AB�R���R��p�B��f                                    Bxm�4�  T          AW33�XQ�Az����$ffB�  �XQ�AB�H��Q���G�B�                                      Bxm�C>  �          AU�j�HA����B��H�j�HAC\)�p  ��33BՀ                                     Bxm�Q�  �          AV{�Dz�A(��ff�,�RBأ��Dz�A@����{��p�BЏ\                                    Bxm�`�  
�          AU���Z=qA=q�	G��   B��f�Z=qAB�H������p�B�=q                                    Bxm�o0  �          AX���P  A&�H� (��
=B֔{�P  AK
=�]p��l��Bг3                                    Bxm�}�  �          AXz��FffA\)�	p���HB֏\�FffAG�����(�B��f                                    Bxm�|  �          AS��<(�A
=��
��HB�Ǯ�<(�AB�H��ff��(�B�(�                                    Bxm�"  �          AQ���^�RA)p�������ffB�aH�^�RAE���!G�B�p�                                    Bxm��  �          AQ�^�RA-����  ��B׏\�^�RAH  ��Q��(�B��                                    Bxm�n  h          APz��?\)A4  ��33��\)B�Ǯ�?\)AI녿�Q����BΨ�                                    Bxm��  T          AR�\�>{A3\)��
=�ᙚBѸR�>{AK��Ǯ���B�Q�                                    Bxm�պ  �          AT���UA2ff��\)����B�\)�UAL(���=q���B�W
                                    Bxm��`  �          AO\)��A5����\)��ffB�=q��AJ�R�������B��)                                    Bxm��  
�          ATz��HQ�AA��G�����B��HQ�AMp�?�@�
B�p�                                    Bxm��  h          AS
=�dz�A<�����
��  B՞��dz�AJ�H=�?�\B�p�                                    Bxm�R  
Z          A\���w
=AF�H���H��G�B�z��w
=AS�
>�p�?\B�z�                                    Bxm��  
�          A]��n�RAK���  ��ffBԨ��n�RAT��?�G�@�p�B�L�                                    Bxm�-�  
�          A^{�N�RAQG��S�
�\��B��N�RAU�?��
@�B�.                                    Bxm�<D  T          A\  �S�
AO��J�H�T��BО��S�
AS�?���@�  B��                                    Bxm�J�  �          A[�
�[�AMG��dz��p��B��f�[�AS�
?�@�B�                                    Bxm�Y�  T          A\���EAO��fff�q�B����EAV{?�
=@�ffB�                                      Bxm�h6  �          AY���@  AM�P���]�B�G��@  AR�\?ٙ�@�Bͳ3                                    Bxm�v�  T          AV=q��HAJ�R�^{�p��Bɨ���HAP��?�
=@���B�                                    Bxmꅂ  �          AM�
=qAFff�=q�-B�Ǯ�
=qAF�R@33A%G�B�                                    Bxm�(  T          AI��#33@أ��"{�Y��B�B��#33A (���{��BД{                                    Bxm��  �          A;�
�[�Ap���ff�  B�ff�[�A*�H�@���pQ�Bי�                                    Bxm�t  T          AHQ��p  A-���G���  B�=q�p  A>�R�@  �Z�HB���                                    Bxm��  �          AEG��XQ�A-����{��33B֞��XQ�A<�þ�녿�
=B��                                    Bxm���            AJ�\�(Q�A;��q����B�Q��(Q�AD��?=p�@W
=B�.                                    Bxm��f  �          AN�R�K�AD���\)�   B��K�ADQ�@�A)��B�{                                    Bxm��  �          AN�R�mp�AEp��p�����B�p��mp�A=��@fffA�G�B�                                    Bxm���  �          AXQ��_\)AO�?Tz�@c33B���_\)A<��@�
=A��B���                                    Bxm�	X  �          AU��hQ�AM��?z�@�RBӅ�hQ�A<z�@�A�(�B�=q                                    Bxm��  
�          AQG��8��AJff���
���
Bͳ3�8��ADQ�@W
=Ao33B΀                                     Bxm�&�  
�          AV�\�G
=AO\)�G��UB�  �G
=AE�@{�A�
=B�8R                                    Bxm�5J  T          AW\)�/\)AP��?�ff@�p�B��/\)A8z�@˅AᙚB�                                    Bxm�C�  �          AT(��=qAN=q?���A33B�.�=qA5p�@�A�Q�B�                                      Bxm�R�  �          AX���!G�AT��?
=q@�\B�p��!G�AC�@�  A��RB�G�                                    Bxm�a<  T          AX  �!G�AT  >���?�G�BɊ=�!G�ADQ�@�Q�A�=qB�33                                    Bxm�o�  �          AF{��\A?�?��AQ�Bɮ��\A(Q�@���A�B̔{                                    Bxm�~�  �          AQG���Q�A.�H��(���
=B�ff��Q�A=��G���Q�B���                                    Bxm�.  �          A\z��˅A+�
��������B�B��˅AAp�������p�B�z�                                    Bxm��  |          AY���
=A3���p����RB���
=AEG��\(��l(�B��f                                    Bxm�z  �          AB�\��33A'����\��=qB����33A3�
��G���B�Q�                                    Bxm�   _          A@�����\AQ������B�q���\A,�׿�{����B�33                                    Bxm���  �          A>ff��  A���z��\)B����  A!��c33����B螸                                    Bxm��l            AHz�����A  ���R��G�B�
=����A&�H���(��B��f                                    Bxm��  �          AG
=����A7\)��  ��G�B�.����A2�R@6ffAT��B�33                                    Bxm��  �          AK\)�r�\AB{?:�H@R�\BָR�r�\A1��@�{A�z�BٸR                                    Bxm�^  
�          AL����Q�AA��?�\)@�  Bب���Q�A/33@���A��HB�=q                                    Bxm�  
�          AN�R�s33ADQ�?��@��B�k��s33A0��@�Q�A�  B���                                    Bxm��  �          AM�EAC�
@�\A$z�B�Q��EA*�R@��A��
B�G�                                    Bxm�.P  T          AI�����HA8��@�
=A��B�aH���HA�@�
=B 33BŮ                                    Bxm�<�  �          AH  �\)A1�@��
A�z�B�ff�\)A�A{B<�B��
                                    Bxm�K�  �          AD��@   A�H@�=qBffB�z�@   @��A��B_p�B�Ǯ                                    Bxm�ZB  
w          A:�H@�@��A��B>�HBe�@�@L��A$��B{�RB�                                    Bxm�h�  ^          AQ���
=qAAp�@��RA�ffB���
=qA{AG�B)�HB�k�                                    Bxm�w�  �          AU��>B�\AJ{@�A�z�B�G�>B�\A&�RAz�B�HB��f                                    Bxm�4  J          AR=q>�AJ=q@b�\AzffB���>�A*ff@�B  B���                                    Bxm��  ^          AT��>aG�AO\)@>{ANffB�>aG�A2�\@�B�B��3                                    Bxm죀  T          AT(���Q�AP��@z�A!p�B�(���Q�A7\)@���A�
=B���                                    Bxm�&  J          AU���Q�APQ�@(Q�A6�HB���Q�A5p�@�A��B��H                                    Bxm���  ,          AV�H�aG�AT��?�@���B��\�aG�A>�R@ÅA؏\B�k�                                    Bxm��r  
�          AT�Ϳ�{AR=q?˅@��
B�����{A<��@�
=A��
B��3                                    Bxm��  |          AI����ffAEp�?��A	p�B�W
��ffA/\)@��RA�z�B��                                    Bxm��  ,          AD�׿�A?
=@\)A;33B�����A&=q@���A��BÞ�                                    Bxm��d  �          A=�У�A1�@xQ�A�
=B��ͿУ�A��@��B��B�\)                                    Bxm�

  �          A9���z�A-p�@c33A���Bɏ\�z�A  @�Q�BB��
                                    Bxm��  T          A:�R��
A3\)@\)AD��B�Ǯ��
A\)@�(�A��B��f                                    Bxm�'V  
�          AJff��  A>{@dz�A�G�B��׿�  A�@�B�B�z�                                    Bxm�5�  "          AZ�R�=p�AG33@�33A���B�Ǯ�=p�AG�A\)B/G�B��\                                    Bxm�D�  ,          A\  �У�AF{@�  A�B�  �У�A�A�B0�B�                                    Bxm�SH  T          A]�h��A8��@�\BB��h��Ap�A0(�BR��B�z�                                    Bxm�a�  T          AJ{?��
@ǮAz�Ba�B�=q?��
@,(�A1�B���B��                                    Bxm�p�  �          AA녿}p�A�H@ҏ\Bz�B�W
�}p�@�p�AQ�BS  Bî                                    Bxm�:  |          A>ff��\A��@�  B�HB�p���\@\A!Bk��B���                                    Bxm��  T          AJ=q=��
AG�A
�HB1��B��=��
@�{A1�B�RB�G�                                    Bxm휆  �          AMG����
A�A ��B G�B��쾣�
@�33A,z�Bm�
B�                                    Bxm��,  T          A0���#�
A)p�?�
=A  B�8R�#�
A�H@��HA�(�B�ff                                    Bxm���  �          A(  �-p�A�
?��
Az�B�W
�-p�Ap�@��A���B�{                                    Bxm��x  
�          A z���A�Ϳ����(Q�B�p���A?�
=A�
B�B�                                    Bxm��  �          A!��;�Aff��\�7�B����;�Az�@-p�Ax��B�{                                    Bxm���  	�          A*{�]p�A�\?�
=@���Bڀ �]p�A=q@��
AΣ�Bޏ\                                    Bxm��j  T          A.�H�S33A"=q?�A=qB�\�S33Az�@��A�Q�B�(�                                    Bxm�  
�          A/
=�6ffA&�\?��HA�Bң��6ffAz�@���A�
=B�33                                    Bxm��  T          A;33�N{A3
=?:�H@e�B�\�N{A%G�@�33A�33B֏\                                    Bxm� \  �          A<(��:�HA5��R�AG�B����:�HA/
=@HQ�Aw
=B��H                                    Bxm�/  T          A6{�>�RA/���(��
�HB�aH�>�RA((�@J�HA�p�Bӣ�                                    Bxm�=�  �          A5��EA-p�<�>#�
B�Ǯ�EA#�
@c�
A�G�BՊ=                                    Bxm�LN            A��C33A33?5@��HB����C33A\)@o\)A��\Bڨ�                                    Bxm�Z�  
�          A"�R�]p�A�
?�@L��B�\�]p�A��@g
=A�\)B���                                    Bxm�i�  
�          A!��y��A33?z�H@��B�{�y��@�@j=qA�(�B��H                                    Bxm�x@  "          A���  A�@\)A��\BҞ��  @�
=@�Q�B	��B�Ǯ                                    Bxm��  |          A�R�&ffA�@�RAb{B�
=�&ff@��@�33B {B��                                    Bxm  T          A:ff�O\)A/�@��A.�\B���O\)A�R@��
A�RB�{                                    Bxm�2  "          AE�����A9���Ǯ��B�
=���A1@W
=A}G�Bܣ�                                    Bxm��  
�          A:�\�0��@�z�Ap�B�\B��H�0��?�p�A,��B��B�=q                                    Bxm��~  �          AA�@���Y��A2ffB�\C��
@����G�A�BS  C���                                    Bxm��$  T          A<z�@[�����A�HBN��C�R@[��  @��B
�\C�o\                                    Bxm���  T          AI�@���ȣ�A�\BM{C��@���Q�@��B=qC�ff                                    Bxm��p  J          AJ�\?���޸RAEG�B��C��3?����=qA2{By{C��=                                    Bxm��  
2          AEp�?�=q�?\)A=�B�(�C��
?�=q�ӅA$��Bc33C��                                    Bxm�
�  �          AG33@���QG�A<��B��qC�Q�@�����HA"=qBZz�C�"�                                    Bxm�b  �          AG\)@'
=�r�\A9p�B���C�G�@'
=���A��BN��C��                                    Bxm�(  "          AG
=@*�H��=qA2=qB~��C�y�@*�H���A��B:�
C�q                                    Bxm�6�  
�          AD��@$z�����A(z�BkC���@$z���A�RB&z�C�"�                                    Bxm�ET  
�          AN�H@!G�����A8��B~z�C�f@!G��Q�Ap�B9�RC�9�                                    Bxm�S�  �          AK\)?��H��z�A:�\B�L�C�w
?��H��ABD�RC���                                    Bxm�b�  
�          AC�?+���\)A1B�L�C��
?+���A�
B=�
C�W
                                    Bxm�qF  �          A=@ ������A-G�B��fC���@ ����\A�
BG�C���                                    Bxm��  �          A8��@{�aG�A+33B�8RC��H@{��{A��BO33C��                                    Bxm  �          A:=q@
=�\��A.ffB�\)C��R@
=��Az�BS�C��H                                    Bxm�8  �          AJ�R@Mp���\A
=BL�C�Ф@Mp�� ��@��B��C���                                    Bxm��  T          AZ�R@dz��G�A#�
BA��C��q@dz��1��@�z�A��C���                                    Bxmﺄ  �          AT��@XQ���
A
=B:�C��q@XQ��1G�@�=qA��C�~�                                    Bxm��*  
�          AR�R@S33��HA(�B9�C��{@S33�/\)@�A�RC�c�                                    Bxm���  �          AT��@W
=��
A
=B4��C��@W
=�3�@���A�G�C�T{                                    Bxm��v  
�          A>�\@QG���@�  B	Q�C���@QG��/33@b�\A��C�P�                                    Bxm��  T          AQ�?�����@�{B��C�4{?��\)@k�A��C��{                                    Bxm��  �          A�H=��
��\)AQ�B��RC��{=��
�z�H@�B{��C���                                    Bxm�h  "          A5p��#33@�A�BJ��BݸR�#33@k�A%��B�B�B�\)                                    Bxm�!  
�          A'����@�Q�@��B6�B�
=���@��RA�
Bx��B�.                                    Bxm�/�  �          Az��@�G�����bffB��=��@ٙ�����\)B�#�                                    Bxm�>Z            A-�J=q@��H��  �.�B�녿J=qA33���H�Ώ\B���                                    Bxm�M   "          AK�
��=qAJff�@  �W
=B��Ὺ=qADQ�@H��Ad��B�W
                                    Bxm�[�  �          AYG����AW33���H��(�B��쿱�AR�H@<(�AH  B�\                                    Bxm�jL            AV=q��(�AR�R��\)��
=B��)��(�AP��@p�A)�B�                                      Bxm�x�  ^          APzῐ��AJ�R�'
=�9p�B�33����AM�?�{@�  B�
=                                    Bxm���  J          AK��k�AFff����  B�� �k�AG\)?�(�@�=qB�p�                                    Bxm�>  
�          A>�H�\(�A0��@�\)A�33B��\(�A�
@�B�B���                                    Bxm��  �          AA녿�\)A%��@��A�33B�\��\)A ��AffB>{B�Q�                                    Bxm�  T          AM��A&�R@���B	��Bʞ���@�  A!G�BM��B��                                    Bxm��0  
Z          APz���
A-�@ᙚB��Bƣ׿��
A33A�BG�RB�p�                                    Bxm���  
�          AN�R�@  A>�\@�  A��B�=q�@  A
=AB�B��                                    Bxm��|  T          AF�R���A5�@��\A�=qB��Ή�A��A   B!p�B�G�                                    Bxm��"  
�          AA녿�=qA(Q�@��A���B����=qA�A
{B6�BɊ=                                    Bxm���  T          A?33�\A"�R@�=qA�Q�B���\@���AG�B>�B�                                    Bxm�n  T          A?��ffA$��@�z�A��
B�{�ffA=qA33B4�B��                                    Bxm�  T          AE���HA#�@ʏ\A���BΨ���H@��
Ap�B?G�B�33                                    Bxm�(�  
�          AK
=�c33A�A��B6�B�8R�c33@��A0��Bs(�B�
=                                    Bxm�7`  
�          AM�s�
@���A.ffBi��B�#��s�
@�\A@��B��{C�=                                    Bxm�F  "          AQp�����@���A,��B^p�B�������@$z�AA��B�8RC�                                    Bxm�T�  �          AW��L��@�
=A�BE(�B�Ǯ�L��@��A>=qB��B��                                    Bxm�cR  �          AXQ��'�AA=qB/��B�B��'�@�A;�Bp�HB��                                    Bxm�q�  �          ANff�Q�A�RAffB<(�BӀ �Q�@���A7\)B|B�p�                                    Bxm�  �          AR�R�&ffA"�\@��B�Bнq�&ff@��
A)�BV��B��                                    Bxm�D  �          AU���@  A0��@�(�A�B�L��@  A��A�B=(�Bڙ�                                    Bxm��  �          AX���U�A<Q�@�G�A���Bӊ=�U�A��A�
B&p�B�\)                                    Bxm�  �          AX  �`  A@Q�@��HA���B�k��`  A (�A{BG�Bڀ                                     Bxm�6  T          AV�R�z�HA@��@�(�A�z�B�  �z�HA$  @��RBG�B���                                    Bxm���  �          AW��c33AE�@w
=A�p�B��c33A+\)@�G�B�Bخ                                    Bxm�؂  r          A*{�$z�A��@:�HA��\B�k��$z�Az�@�z�Bp�BՔ{                                    Bxm��(  
�          A=q�#�
A�H@>{A���BָR�#�
@�
=@�
=BffB�L�                                    Bxm���  
�          A)��>�RA�@�
=B��B�p��>�R@��@�z�BB  B枸                                    Bxm�t  6          A/��?\)A��@���A��Bڙ��?\)@ҏ\@�Q�B<�B��                                    Bxm�  
�          @�p���p�@��@ffA�p�B�녿�p�@�Q�@hQ�Bz�B�.                                    Bxm�!�  J          A#�
�{@��H?@  @�=qB�33�{@�z�@�A���B➸                                    Bxm�0f  �          A'�@�����\)���R�
=C�^�@�����ff��G��@��C��H                                    Bxm�?  
�          A)@��R��(���33�\)C��f@��R��Q�� ���F(�C�
                                    Bxm�M�  �          A,(�@����\)��  �  C�� @����33� (��A{C���                                    Bxm�\X  �          A*�R@�Q���p��ə���C�H�@�Q��w���\)�9�RC�"�                                    Bxm�j�  �          A((�@�p���(����
��C�^�@�p��h����
=�5p�C�4{                                    Bxm�y�  
�          A&�H@ٙ���G���z��
=C�j=@ٙ��l���߮�(\)C��)                                    Bxm�J  "          A�@��s33����	�RC�+�@���\�θR�"�C�"�                                    Bxm��  
�          A ��A(��k������33C��A(�>�ff��\)��R@;�                                    Bxm�  
�          A.ff@��U���=q�+
=C���@���{���R�>z�C���                                    Bxm�<  �          A5�@�=q���R����L�\C���@�=q�������rQ�C�                                    Bxm���  �          A7\)@����Q��z��XffC�j=@����Q��$  �}�
C��)                                    Bxm�ш  
Z          A5�@��H��=q�=q�TG�C��R@��H�!G��$Q����C��                                    Bxm��.  
�          A:�H@�=q���
����\=qC�  @�=q��p��'���RC�!H                                    Bxm���  �          A<��@���������R33C�{@���:�H�*{W
C��
                                    Bxm��z  �          A:=q@o\)������J��C�O\@o\)�a��&=q�~�C�O\                                    Bxm�   �          A"�\@*=q������
�Bz�C���@*=q�qG�����{�
C���                                    Bxm��  �          A3
=@c33�\�33�L
=C�#�@c33�Y��� ����
C�#�                                    Bxm�)l  
�          A5p�@c33������
�bp�C�H�@c33��\�(��aHC��)                                    Bxm�8  �          A/\)@�  ����\�c�C�
=@�  �ٙ�� z��C��                                    Bxm�F�  T          A-��@�Ϳ!G��G�ffC�� @��@z��{=qB �\                                    Bxm�U^  "          A!��q�A33�,(��w�B����q�A����Q�   B�\                                    Bxm�d  �          A\)�qG�@�������
�
B���qG�AG��W
=��p�B��                                    Bxm�r�  �          A&�\�}p�@ᙚ��G���HB�3�}p�A	p���=q�ď\B�=                                    Bxm�P  T          A"�H�fff@�(���=q�(ffB����fffA(����R�ܸRB�{                                    Bxm��  �          A$(��P  @�  ���;
=B�
=�P  AG���
=� z�B��f                                    Bxm�  �          A%G��8��@���� (��L(�B����8��@�G���33�
=Bܳ3                                    Bxm�B  
Z          A�\�*�H@�ff����Lz�B�B��*�H@����\)�Q�B�#�                                    Bxm��  �          A&=q�5@��\��H�Y��B�\�5@�ff��(���Bݳ3                                    Bxm�ʎ  �          A&=q�P��@g��{�w�C\�P��@�������C
=B�(�                                    Bxm��4  �          A'����׾�=q��k�C7�)����@ff��H�x=qC33                                    Bxm���  T          A"�R��\)?n{�  �~�\C'����\)@S33��\�b��C\                                    Bxm���  
�          A�����@h�����}��B�� ���@�z���z��E=qB��                                    Bxm�&  
�          A!��A�@Z=q�\)�|�C�H�A�@�\)���GffB�R                                    Bxm��  �          A=q�-p�@G��\)C)�-p�@�ff��
�b��B�                                      Bxm�"r  T          A=q�z�?Y���(�(�C�=�z�@Vff�
�H�qB�\)                                    Bxm�1  �          @�  �?�
=�����C33�@[���{�_��B�R                                    Bxm�?�  �          @��ÿ���=p����
 �CV޸���?�{��G�.C �                                    Bxm�Nd  �          @���@�{@z=qB33B�����@{�@�p�BQBĞ�                                    Bxm�]
  "          A�\�B�\@˅@�\)B#
=B�aH�B�\@�(�@�  Ba��B���                                    Bxm�k�  �          AQ��@�=q���
�2�\B��)��@�ff?z�@�ffB�p�                                    Bxm�zV  �          A�\�Aff�7
=��(�Bх�Ap���33�z�B�.                                    Bxm��  �          A����A(��(��Z�RB�B����AQ�>��
@   B�.                                    Bxm���  T          A"�\�z�A���^{��z�B��)�z�A
=�Q����HB�G�                                    Bxm��H  "          A)p����A\)�HQ����B�W
���A'
=���ÿ�\B�aH                                    Bxm���  �          A,(���  A"�H�U��\)B��
��  A+33���$z�B��3                                    Bxm�Ô  
�          A+33��A"=q�S33���B�  ��A*ff��ff��B��R                                    Bxm��:  T          A0z�h��A���������B��h��A)��1��iB�Ǯ                                    Bxm���  T          A.�R��p�A
=���
���B�B���p�A%�<���{
=B��                                    Bxm��  �          A(  ��{@����Q��&p�B�8R��{Aff������
B��                                     Bxm��,  �          A=q�L��@*�H���\B�G��L��@��H��ff�b��Bƽq                                    Bxm��  
�          Az��R@>�R����aHB˅��R@�ff��
=�U��B�G�                                    Bxm�x  "          A��0��@��H��=q�Q�B��H�0��@��
�C33�îB�u�                                    Bxm�*  
�          A:�H��ffA�H@j=qA�Q�B�\��ff@��
@�
=A�=qC 5�                                    Bxm�8�  �          A?\)��z�A@\(�A�\)B��3��z�A@���A�CL�                                    Bxm�Gj  �          A<����\)A�H@I��Aw\)B�� ��\)A (�@�
=A���C�                                    Bxm�V  
�          A2�H���A�
@<��At��B��3���@���@��A��C��                                    Bxm�d�  
�          A6�\����A��@hQ�A�Q�B�����@���@�Q�A�z�Cff                                    Bxm�s\  
�          A4z���(�A�\@h��A��\C!H��(�@��@��A�G�C޸                                    Bxm��  
�          AA����A�R@��A�  C\)���@�
=@�B(�C�                                    Bxm���  
�          ADz���p�A{@��HA�{CW
��p�@�  @��BQ�C��                                    Bxm��N  
�          A?33���H@�(�@�{AͮC�����H@Ǯ@�\B=qC�H                                    Bxm���  T          A=��Q�A�R@�G�A�(�C����Q�@Ϯ@�  B��C&f                                    Bxm���  �          A=G��ٙ�@�z�@��A�(�C���ٙ�@�=q@�ffB"�Cs3                                    Bxm��@  �          A8Q�����@�{@���A���C������@�33@���B)C
�=                                    Bxm���  
�          A0z��\@أ�@�\)B	��C��\@���@�G�B3�C#�                                    Bxm��  	�          A6�\��
=@�Q�@�33A��HB����
=@��R@�B)33C�                                    Bxm��2  
�          A0�����A\)@\)A�33B������Ap�@�=qBz�B��
                                    Bxm��  �          A3�
��\)A{@�33A��
B�q��\)@�@��B�HB�\                                    Bxm�~  
�          A7����\A
�R@�{A�\B�=q���\@�z�@�  B*��B�\                                    Bxm�#$  "          AB�R��{A(�@�p�A�B�(���{@��HA�B-�B��                                    Bxm�1�  "          A@(���33A��@��B�HB�����33@�=qA�B:��B�\                                    Bxm�@p  
�          A:�H��=qA	�@��B(�B��
��=q@љ�A
ffBA(�B�                                    Bxm�O  �          A(z�����@�Q�@��HBG�B�Q�����@�p�@��\BA�\B���                                    Bxm�]�  
Z          @�z�=�\)@��׿���*�HB��==�\)@��>�G�@��\B��\                                    Bxm�lb            A33��ffA  >#�
?���B̏\��ff@�@�\A{\)Bͅ                                    Bxm�{  
�          A��\)A\)?�@Mp�B�𤿯\)A(�@8Q�A�{B���                                    Bxm���  
�          A#
=���\A!p�?z�@Q�B�ff���\Ap�@J�HA��HB��                                    Bxm��T  T          Az῞�RA=q?:�H@�z�B�녿��RAff@@  A�p�B��
                                    Bxm���  T          Azῢ�\A�R=���?��B����\A�@%Av=qBÔ{                                    Bxm���  T          A�
�\A(�?&ff@p��Bų3�\A(�@H��A��
Bƞ�                                    Bxm��F  
�          A%�ffA �ÿ��\���Bˏ\�ffA ��?�{@��B˙�                                    Bxm���  "          A$�׿��HA!���z���
=B�=q���HA z�?��HA
=B�Q�                                    Bxm��  
�          A$�׿�\)A!G��������
B�p���\)A!p�?��@�B�p�                                    Bxm��8  "          A���  A33�n{����B�{��  A�?��
A�\B�33                                    Bxm���  
1          Az��  A�
�u��33B�
=�  A\)@\)A`Q�B��                                    Bxm��  T          A�׿�(�A����H����B�#׿�(�A�?�Q�@�ffB�#�                                    Bxm�*  T          AG���{A�R���L��B��H��{A�\>�z�?�G�B�aH                                    Bxm�*�  
�          A(���Q�A{��c�B��
��Q�A�H=�\)>�(�B�aH                                    Bxm�9v  ]          A��xQ�A{�z��h��B�� �xQ�A�H<��
=�B��                                    Bxm�H  
�          A(��!G�AG���p��;
=B��{�!G�A�
>�ff@@��B�k�                                    Bxm�V�  +          A�R��{A
�\���
�9�B�zᾮ{AG�>�@FffB�ff                                    Bxm�eh  "          A
ff���A���p���Q�B�  ���A
ff��\)��=qB��                                    Bxm�t  �          AQ��A
�H���tz�B�aH��A  ��Q�
=B�\)                                    Bxm���  �          A�    Az��U���RB�    Ap���G����B�                                    Bxm��Z  �          A녿�Q�A=q��\)��G�Bę���Q�A��!G��v{B��f                                    Bxm��   T          A��>�
=@�����=q�܏\B���>�
=A�ÿ�(��O
=B�aH                                    Bxm���  
�          A�u@���G���
B��;uA\)�P������B�L�                                    Bxm��L  
�          A33��\)@�(���{�33Bͨ���\)Aff�|(����B��f                                    Bxm���  }          A\)��z�@�p������
=B��ÿ�z�@�33�j=q��
=B�\)                                    Bxm�ژ  +          A녿�Q�@��
��ff�,  B����Q�@��
��G�����BθR                                    Bxm��>  �          A33� ��@�=q�����8Q�B�p�� ��@�(����
�
=Bսq                                    Bxm���  "          AzΎ�@�������`�
B��
���@�Q�����(p�B�33                                    Bxm��  T          A�
����@������nz�Bٳ3����@�Q������6z�B�                                    Bxm�0  �          A (�����@E���ff�)B�aH����@�G���{�L��Bӊ=                                    Bxm�#�  �          AG���  @W
=�����{��B��ÿ�  @����=q�F  B�                                    Bxm�2|  
�          A�\��ff@<����B���ff@�Q���(��S��B��                                    Bxm�A"  �          A�>W
=��Q�����(�HC��>W
=��=q��33�b�
C�=q                                    Bxm�O�  �          @���>L�����e��ᙚC�ٚ>L����=q��z��*�RC�                                    Bxm�^n  }          A{?+��������H��HC���?+���\)��z��;p�C�z�                                    Bxm�m  
�          A
=?L����(��{���G�C�{?L����������)�C��\                                    Bxm�{�  T          @�
=�Q���z�?�=qA��C���Q���p��L���\C���                                    Bxm��`  T          AG�=#�
�ff���R�%��C�#�=#�
����^{��Q�C�'�                                    Bxm��  
�          A	G�?�z����z��_�C�f?�z���ff������  C�o\                                    Bxm���  T          A
�\?�
=�����Q��4��C��q?�
=����n{���
C�J=                                    Bxm��R  �          A
ff?Ǯ��������*�\C�L�?Ǯ��{�hQ���ffC���                                    Bxm���  T          A	��?����ٙ��6�HC�4{?���G��l(��ˮC���                                    Bxm�Ӟ  T          A	@   � �Ϳ�(��8��C��q@   ���
�j�H���
C�Z�                                    Bxm��D            A	�@!G�� �ÿ�z��1��C��@!G���z��fff��(�C�g�                                    Bxm���  
�          A�R@&ff����=q�=p�C���@&ff��33�u����C�t{                                    Bxm���  }          A�\@C33�
ff��(��B{C���@C33��z��������HC���                                    Bxm�6  �          AQ�@i��������7�C�4{@i��������\��33C�"�                                    Bxm��  
�          A"=q@P���z�c�
����C�p�@P���Q��N{��ffC��                                    Bxm�+�  �          A((�@��\�������	�C�� @��\�ff�p�����C��{                                    Bxm�:(  �          A((�@`  ���O\)��Q�C�H�@`  ����ff���RC���                                    Bxm�H�  
�          A*ff@w��(�������C��@w��  ��ff����C��                                     Bxm�Wt  �          A+\)@�
=����������C��f@�
=��R�p����\)C�Q�                                    Bxm�f  T          A*{@�z���Ϳ�(���
=C�Z�@�z��=q�q���
=C�&f                                    Bxm�t�  �          A/�@�G�����=q�(�C���@�G��z��\)��z�C�g�                                    Bxm��f  T          A,z�@�p��z��
=��C�\@�p����~{����C��{                                    Bxm��  �          A+�@����{�u����C��@��������R{C�E                                    Bxm���  �          A(�@��H���@\��A��C�=q@��H� z�?\AG�C�.                                    Bxm��X  T          Az�@����p�@(�Ao�
C�Q�@��� z�>�@9��C��R                                    Bxm���  T          Aff@~�R��H@�
AL  C���@~�R�
=�#�
��Q�C��H                                    Bxm�̤  T          AQ�@G
=���?�33A\)C�o\@G
=���#�
�p��C�P�                                    Bxm��J  �          A��@7
=�=q@{AV{C��=@7
=��\���8Q�C��=                                    Bxm���  �          Ap�@%��H@
�HAR�RC�{@%�
=���
���HC��q                                    Bxm���  
�          AG�@(��Q�?z�H@�(�C��@(���������C���                                    Bxm�<  "          A=q?�p���R?��
A�C��{?�p��(��E���{C��=                                    Bxm��  "          A�R?p�����?��
@�33C�˅?p���Q쿬��� (�C��                                    Bxm�$�  �          A
=?�  �  ?333@�C�xR?�  �{��33��C��f                                    Bxm�3.  �          A33�h����\)@<��A��C�&f�h����  ?�(�A
=C�k�                                    Bxm�A�  �          AQ�?����������\�(�C�>�?�����Q���z��G�\C�                                      Bxm�Pz  T          A>�Q�����E��33C�<)>�Q����������
(�C�g�                                    Bxm�_   �          A�\?E��
=q�5����C��=?E������� �
C�ٚ                                    Bxm�m�            A�
?B�\�
=�l(���=qC���?B�\������z��ffC��R                                    Bxm�|l  
�          A\)>Ǯ����~{�̣�C�W
>Ǯ��=q��(����C���                                    Bxm��  T          A\)���녿����ffC������(��j=q����C���                                    Bxm���  "          A�H��G���ÿ�G���C��ͽ�G���\�s33���RC���                                    Bxm��^  T          A\)�B�\�Q��Q��!G�C�p��B�\���}p����C�e                                    Bxm��  T          A�\�G��33>���@��C�� �G��(���z��9�C��{                                    Bxm�Ū  �          A�\���\��?�@ٙ�C�J=���\�����R��{C�H�                                    Bxm��P  T          A(���z���R?&ff@uC��ÿ�z��z��(��"�\C��=                                    Bxm���  "          AQ��(��Q�?O\)@�z�C���(��
=��p���C��
                                    Bxm��  �          A �Ϳ���R?�=qA((�C�aH���G��z��Tz�C�w
                                    Bxm� B  
�          A�R�ٙ�����\)����C�
=�ٙ���(Q��w�C���                                    Bxm��  "          Aff�
=q�=q?!G�@j=qC����
=q�  ��p��!G�C���                                    Bxm��  �          A=q����
?�33@�ffC��������Q���{C��                                    Bxm�,4  T          A!���6ff�=q?��R@ᙚC}���6ff��\�����ffC}�\                                    Bxm�:�  �          A(  �U��=q?xQ�@���C{c��U��G���(�� z�C{G�                                    Bxm�I�  �          A z���=q���?��@���Cu����=q��ÿ������Cu�\                                    Bxm�X&  �          AG�����Q�@�\AYp�Ct8R�����>��?aG�Ct�q                                    Bxm�f�  L          A33�e���p�?�33AE��Cu���e��=q���
�   CvG�                                    Bxm�ur  *          A(��p  �	�@(Q�A{33CvaH�p  �33>��@1G�CwG�                                    Bxm��  
Z          A33����;��0  C��{����\�,����p�C�O\                                    Bxm���  T          A\)��z��Q��2�\��p�C�3��z�����z���C��R                                    Bxm��d  "          A  �#�
�ff�L����=qC��R�#�
��33��Q��G�C��R                                    Bxm��
  �          Az�@ff��Q�����
�C��@ff��\)��=q�@�\C��                                    Bxm���  T          A��@"�\�����ff��C�q�@"�\��  ��=q�IG�C��                                    Bxm��V  	�          AG�@G���  ������C�l�@G���{��p��A�RC�w
                                    Bxm���  �          A��@*�H���������C�5�@*�H���H��
=�EffC���                                    Bxm��  �          Aff@*�H��Q��������C��@*�H��z����F��C���                                    Bxm��H  "          A�@L����\)��{��C�"�@L�����H��\�N�C���                                    Bxm��  
�          A�@�
��p����H�G�C��f@�
��������G
=C�s3                                    Bxm��  �          A�R?������R������C�Q�?�����Q�����MffC��                                    Bxm�%:  �          A�?Ǯ��(������
�C��f?Ǯ���H���H�A��C�.                                    Bxm�3�  
�          AG�?Ǯ���������	�HC���?Ǯ������ff�@�\C�B�                                    Bxm�B�  T          A�?�����z����R�Q�C��?����ƸR��p��9\)C�C�                                    Bxm�Q,  �          A��?k���\��33���C�w
?k����љ��9ffC�=q                                    Bxm�_�  
�          A�
@G������(���
C��H@G����
��
=�M\)C��)                                    Bxm�nx  "          A=q?�{�޸R��Q���C�w
?�{��\)��G��F=qC�\)                                    Bxm�}  "          AG������
�h���ÅC���������8Q���
=C~+�                                    Bxm���  
Z          Az���=q���?���AC
=Cq����=q���ü��W
=Cr^�                                    Bxm��j  "          A  ������?��A=qCq=q����������H�L(�Cq�
                                    Bxm��  
�          A{��G���?�(�@�\)Cp�\��G���R�G���G�Cp�R                                    Bxm���  "          A�
��
=��
>k�?�33Cs�\��
=��׿�=q�2=qCs                                    Bxm��\  �          A=q�����33���
��33Cq+��������
=�bffCp33                                    Bxm��  T          A�������\�������Cr=q�������H�G��d(�CqJ=                                    Bxm��  
(          Ap��������?0��@�(�Cp��������\���R���
Cp�
                                    Bxm��N  
�          A�\���� (�<��
=�Cq+�������׿��B=qCpp�                                    Bxm� �  T          A�R��33�G����H��C��׾�33��G���p��(��C���                                    Bxm��  
Z          A��33�
=�.{��33C�/\�33������{��  C~�=                                    Bxm�@  
�          A�R���R�\)�7���{C�.���R��������ffC~�                                    Bxm�,�  �          Aff�
=��\�'
=���\C}�f�
=��G���Q���  C|{                                    Bxm�;�  
�          A
�H�����׿����CkJ=����Ϯ�>{���Ci(�                                    Bxm�J2  "          A����33�θR�Tz����RCc����33����Q��{�
Cb�                                    Bxm�X�  
�          A
�\�����ff�����  CgaH�����
=��\�>�RCf^�                                    Bxm�g~  �          A	�����ָR��\)�
=ChO\����p��=p���G�Ce�q                                    Bxm�v$  �          A\)��p����O\)��\)Ceٚ��p���G������Cd{                                    Bxm���  "          A	��\)��p��Ǯ�'
=C_�{��\)���Ϳ����C33C^=q                                    Bxm��p  �          A�������ȣ׾�G��>�RCcaH������
=�����S�
Cb                                      Bxm��  �          A	���p����
�
=�|��C_�=��p������   �X(�C]�3                                    Bxm���  
�          A���  ���H�ff�h(�CX����  ��{�Mp���ffCTǮ                                    Bxm��b  
�          A���p����ÿTz����HCZ\��p���p����e��CX�                                    Bxm��  �          A
=��(����;�Q��"�\CT�
��(������ (�CSE                                    Bxm�ܮ  T          A(���p���ff���~{CT���p�������7
=CS#�                                    Bxm��T  ~          A����=q��\)�E���33CRٚ��=q�z=q���
�FffCP�                                    Bxm���  
�          A z�������R���u�CaǮ��������z��\��C`:�                                    Bxm��  �          Ap���  �����(��33CaǮ��  ���(Q����HC_aH                                    Bxm�F  T          A������녿У��,Q�Cf
�����
=�K���33Ccff                                    Bxm�%�  
�          A(��G���  �(���(�C��
�G���ff��Q��
=C�G�                                    Bxm�4�  T          A��@�ff��z��1���=qC���@�ff��  ��(����HC�~�                                    Bxm�C8  	�          A	p�@C�
����p���Q�C�H@C�
��\)��\)��(�C�4{                                    Bxm�Q�  "          Ap�@j�H��
=�aG���{C�xR@j�H��������\C�l�                                    Bxm�`�  �          Aff@�33�����U���\C���@�33���
��\)�\)C��                                    Bxm�o*  
�          A�\@������:�H��
=C�AH@���������H���RC��                                    Bxm�}�  "          A�H@�z����HQ���{C���@�z���(������33C��\                                    Bxm��v  �          A{@��H�ᙚ�333���C���@��H������{��z�C�aH                                    Bxm��  �          A�@�\)�������\����C�f@�\)���
���
�\)C�AH                                    Bxm���  �          A
ff@��\��
=������C��\@��\�i�����H�,(�C�#�                                    Bxm��h  
�          @�33�����θR�Y����33Cn#�������������=qClu�                                    Bxm��  ~          A�����ə�>���@
=Cg�)�����{��(���Cg)                                    Bxm�մ  
�          A Q�������ff>k�?�33Ci� ������녿����\)Ch�f                                    Bxm��Z  
�          @�{�tz���=q�G����Cp�R�tz���p���H����Co:�                                    Bxm��   T          A��W���A�B�G�C<�3�W��)��A�B{  CZ33                                    Bxm��  T          A=q�S33��A��B��CN�
�S33�r�\A�RBi�RCd��                                    Bxm�L  �          A\)�C33�G�Az�B�8RCX���C33��33@��RBa{Cj�q                                    Bxm��  "          A�������E�@�z�BSffCT(��������@���B/�RC`Ǯ                                    Bxm�-�  T          A{�����o\)@�=qB>\)CWff�������@�{B33Ca��                                    Bxm�<>  �          A����33�N{@�33B.�HCO�\��33���@��B��CY�
                                    Bxm�J�  L          A���\)�1G�@�B9��CL����\)��z�@�G�B��CXO\                                    Bxm�Y�  
�          A
=��G�����@�ffB)p�CZ� ��G���33@�p�B�RCb��                                    Bxm�h0  �          A(�����}p�@���BffCVJ=�����(�@�(�A�ffC]z�                                    Bxm�v�  "          A�\��G�����@�B�
CU�3��G���@���A�z�C\�)                                    Bxm��|  "          A�R��(�����@���A�  CZ����(����@K�A���C_�R                                    Bxm��"  �          A(���33�У�@(��A�G�Ca��33��
=?s33@��Cc��                                    Bxm���  
�          Ap����
��z�@�(�A�ffCZ����
��  @/\)A���C_Q�                                    Bxm��n  �          A������@�p�B�CTO\�����H@��A�(�C[�R                                    Bxm��  
(          A  ��  ���\@���A��
CU(���  ����@HQ�A���CZ@                                     Bxm�κ  "          Aff��Q�����@p  A��HCY�3��Q���G�@�Af�\C]�                                     Bxm��`  
Z          A������33@R�\A��HCSǮ������@33AJ�\CW\)                                    Bxm��  
�          A�\��ff���R@^{A�=qCR���ff��@��A_�CV��                                    Bxm���  "          Ap�������z�@^�RA�{CQ^��������
@�\A\z�CUB�                                    Bxn 	R  
�          A�\��ff�|(�@j=qA���CO{��ff��
=@$z�A~=qCS�                                    Bxn �  �          A�R���R��
=@J=qA�=qCP�R���R��(�@ ��AG33CTO\                                    Bxn &�  �          A
�H���\��\)�����p�Ci�����\��ff���R�T��Ch�{                                    Bxn 5D  T          A=q�Fff��{��=q�1�CwaH�Fff��=q�XQ���(�Cu��                                    Bxn C�  �          Ap��aG���������
C��3�aG����H��p���ffC�w
                                    Bxn R�  
�          A
=�o\)��\)��
=� z�Csp��o\)��ff�@�����RCq��                                    Bxn a6  T          A��]p���Q쿆ff��ffCuJ=�]p�����8Q����Cs�                                    Bxn o�  �          A\)�S33��׿�����HCx�)�S33� (��I����  Cw�
                                    Bxn ~�  �          A녿�����8Q���z�C�0�����\�����C��3                                    Bxn �(  �          Az�G��	G���p��  C�b��G���{�dz����C�,�                                    Bxn ��  �          AG�?�  �=q�{��{C�c�?�  ��Q���\)����C���                                    Bxn �t  �          Aff��p���ÿ���*ffC�J=��p����
�n�R�Ə\C��\                                    Bxn �  �          A(������
�R��\)�?\)C��H�������~�R�хC�G�                                    Bxn ��  �          A  ��
=��ÿ�\�0z�C�����
=���~�R��(�C��=                                    Bxn �f  �          A�R��G������
����C�����G��33�dz���ffC���                                    Bxn �  �          A�
���p���{�(�C�b����
=q�y����(�C�AH                                    Bxn �  �          A(�����ff���H��=qC�9��������a����HC�+�                                    BxnX  �          A?�{�z�#�
��\)C�ff?�{����hz�C���                                    Bxn�  �          A�?�p��>.{?��C���?�p�����(��VffC�ٚ                                    Bxn�  �          A�H?xQ��G��aG����\C��{?xQ�����L�����RC�!H                                    Bxn.J  
�          Aff?�ff���>�p�@�\C�7
?�ff�p���p��C�C�H�                                    Bxn<�  �          A(�?��H���?�G�@�  C��{?��H�Q쿧���HC���                                    BxnK�  �          A\)�У���
?��A ��C�ٚ�У��Q쿂�\��33C�޸                                    BxnZ<  �          A{�������?�p�@�G�C�� ���������
��Q�C���                                    Bxnh�  �          Az������?���A(�Cc����33�5��C��                                    Bxnw�  �          A33��\)��R@G
=A���C����\)�
=?p��@���C��                                    Bxn�.  �          A�
�
�H��33@w�A�G�C~�=�
�H�	�?�  A3\)C�=                                    Bxn��  �          Aff�n{���@�\AT��C�⏿n{��;L�Ϳ��C��R                                    Bxn�z  �          Ap���G����>���@$z�C����G��	�������B{C��f                                    Bxn�   �          Aff���
=?Y��@���C�J=���녿��H���C�Ff                                    Bxn��  �          AG���z��@3�
A��\C�׾�z����?
=@u�C��                                    Bxn�l  �          A��=�G��  @q�A�
=C�` =�G��
=?ǮA�C�XR                                    Bxn�  �          A=q?n{��@���B{C��\?n{��\@.{A��RC�&f                                    Bxn�  �          A����
��z�@�(�B	G�C��=���
�ff@8��A�=qC���                                    Bxn�^  �          Az�>����33@�=qB�
C��>����\@E�A�\)C��f                                    Bxn
  �          Ap�?��θR@�{B*
=C�P�?�����@��HA��
C���                                    Bxn�  �          A{?�����{@�G�BJ��C�8R?�����  @�p�BQ�C��q                                    Bxn'P  �          A
{=u���
@�(�BQ\)C�XR=u��z�@��HB�C�C�                                    Bxn5�  �          @�Q��J=q���H@I��A�{Cs���J=q��?�A(��Cu��                                    BxnD�  �          A=q���R���@-p�A�
=Ck�\���R��G�?n{@���Cm�f                                    BxnSB  �          A���33��33?��
A*�RCg@ ��33���þ����ffCh�                                    Bxna�  �          A���G����?�p�A_
=Cf�3��G��θR>��
@��Ch
=                                    Bxnp�  �          A���\)��z�?�
=A<��Cl��\)��33��Q�(��Cl�{                                    Bxn4  �          Ap�������
=?�\)AO\)Ch�������׮>�?k�Ci��                                    Bxn��  T          A�H���H��@'�A�
=Cf�����H����?p��@��
Ch�\                                    Bxn��  T          A�������
=@�A�
Cc33�������
?5@�G�Ce
=                                    Bxn�&  �          A�R��=q����?L��@��RCa�q��=q���ÿJ=q���
Cb                                      Bxn��  �          A�\��
=��G�?�ff@�Cb����
=��33�\)�s33Cb�H                                    Bxn�r  �          A�
��������?�(�A�Cb#�������{����z�HCb�3                                    Bxn�  �          A	��\)��=q?�
=A�C`=q��\)��  ���fffCa�                                    Bxn�  �          A����  ����@J�HA�(�CV���  ��
=?�\)AF=qCY��                                    Bxn�d  �          A�\��Q��8Q�@��RB�RCK�{��Q���z�@���A�{CTh�                                    Bxn
  �          A�
�������?��@�
=CWs3�����(���z���CX                                      Bxn�  �          A\)��33���@p�A�  C[�q��33��{?p��@��C]�q                                    Bxn V  �          A��љ���  @A�A�{C\��љ���33?�p�A�RC^�q                                    Bxn.�  �          A�����@�A\)CWh�����z�?��@ۅCY�H                                    Bxn=�  �          A��p���z�?�\A8��CVE��p���>�(�@1�CWٚ                                    BxnLH  �          A{������  ?���A;�
C\O\������G�>�=q?�(�C]��                                    BxnZ�  �          A�����:=q@L��A��\CI�����i��@�Aw�CM                                    Bxni�  �          A���=q�.�R@VffA��RCG����=q�`��@ ��A�  CL�H                                    Bxnx:  �          A\)��p��Dz�@'
=A��\CI�
��p��i��?��HA6ffCMz�                                    Bxn��  �          A	p�����5�@@  A�
=CH������aG�@��Af�RCM                                      Bxn��  �          Ap���  �Dz�@�AqG�CJ����  �aG�?��RA\)CM��                                    Bxn�,  �          A Q���\)��z῁G����Ci����\)�����*�H���CgG�                                    Bxn��  �          @��Z�H���������Cq��Z�H������Q�����Cn\)                                    Bxn�x  �          @��\�33����l(���  C|�)�33���H�����+33Cx�R                                    Bxn�  �          A\)�,(��Ϯ��������Cwu��,(����
���
�5�Cr:�                                    Bxn��  �          A��g���=q�Q��v�HCr
=�g���Q��w����Cn�                                    Bxn�j  �          A�R�c33��녿�  ��
CsL��c33��\)�E��p�Cq@                                     Bxn�  �          A(��<(������*=q���Cx���<(���G���=q�{CuǮ                                    Bxn
�  T          A�R�r�\��\��Q��>{Cqٚ�r�\���
�aG����CoE                                    Bxn\  �          Aff������\)?W
=@�Q�Cl�H������ff�����  Cl�                                    Bxn(  �          A=q��G���zῙ����Cf\)��G����\�7���  Cc�=                                    Bxn6�  �          A	�������ff��
�]Clu���������vff����Ci!H                                    BxnEN  �          A���p��ָR����
=Cn&f��p���33��  ��
=Cj��                                    BxnS�  �          A{��(���33�u���Cj�3��(����H��=q�R{Ci�)                                    Bxnb�  �          Ap����R��z῝p����Cm����R�ə��B�\��{Cj�                                    Bxnq@  �          @��R��(���(���{��Cn���(������Fff���Ck�\                                    Bxn�  �          A����H�ٙ��\)��G�Co  ���H�����Q���
=Cmn                                    Bxn��  �          A�H��  �޸R<�>aG�Cn����  �ָR�����R{Cm�H                                    Bxn�2  �          A�H���R������Q�Cn�R���R�У���H����Cm#�                                    Bxn��  �          A z�������Ϳ޸R�H��CnG�������^�R��  CkE                                    Bxn�~  �          @�=q�qG����ÿ�33�a�Cp��qG������fff�ܣ�Cl�
                                    Bxn�$  T          @�z��ff�أ��U��  Cz�)�ff��33����
=Cw@                                     Bxn��  �          @��
�L���ڏ\�Q��|(�Ct�f�L������z=q��\)Cq�f                                    Bxn�p  �          @�z��U�ۅ���T��Ct��U�Å�g
=��33CqQ�                                    Bxn�  �          Ap��y���ᙚ>�z�@�Cq\�y���ۅ��33�<z�Cpff                                    Bxn�  �          A�R�y����(�?5@�\)CqT{�y���ᙚ���\�\)Cq\                                    Bxnb  �          @�33��G���G�?�  AO33Ck���G����ý#�
��z�Cl��                                    Bxn!  �          A33�����?�@}p�Cn)�����=q��{���Cm��                                    Bxn/�  �          A����{�љ�?&ff@��\Cjh���{��\)��
=�ffCj�                                    Bxn>T  �          A �����H��G�=�G�?Q�Cmc����H��녿�G��I��Clz�                                    BxnL�  �          @����\�θR>u?��HCj�����\�ȣ׿�ff�5�Ci޸                                    Bxn[�  �          @�\)��=q�˅=�Q�?333Ck�=��=q��z���H(�Cj�
                                    BxnjF  T          @�\�p����G�?0��@�ffCp��p����\)��33��Co��                                    Bxnx�  T          @�����ff�ʏ\>�@dz�Cj����ff�ƸR�����Cjff                                    Bxn��  �          @�=q�o\)��33>�G�@N{CqaH�o\)��ff��p��.�\Cp�)                                    Bxn�8  �          @��\��=q��33>�?xQ�Cj@ ��=q��z����AG�CiO\                                    Bxn��  �          @�����H��z�>�\)@
�HCh{���H�����\)�(��CgW
                                    Bxn��  �          @�(������\?���AG�C]�������R�����\C^aH                                    Bxn�*  �          @���,(���z�#�
�L��Cyc��,(��ۅ�   �s33Cx�)                                    Bxn��  �          @�p��N�R���#�
��{Cu�N�R��z��p��p(�Ct{                                    Bxn�v  �          @�ff������G�>��@<��Ch�q�������Ϳ�\)��Ch�                                    Bxn�  �          @�z�������G�>���@:=qCf��������������Ceu�                                    Bxn��  �          @�����(���ff�&ff���CnT{��(���  � ����{Cl�                                    Bxnh  �          @��������  �#�
��z�Co�����ə�� ����ffCm=q                                    Bxn  �          @�����ff��녿(�����CiW
��ff���
�=q��z�CgT{                                    Bxn(�  �          @����p���  �\(���ffCh
��p���Q��!G���Ce��                                    Bxn7Z  �          @�z���������
=��Cc
�������Q���Q�C`�q                                    BxnF   �          @���  ������R��Cc����  �������f�\Cb                                    BxnT�  �          @���U�θR��ff�"{Cr���U���\�E�����Cp.                                    BxncL  �          @�33��G������n{���Cfu���G������!���ffCc�f                                    Bxnq�  �          @�{������׿�
=�3\)Ce�������(��<(���z�CbW
                                    Bxn��            @����\)�����=q�)p�Ca����\)�����.�R��
=C^5�                                    Bxn�>  �          @�����\���
��  �>�RCZ)���\�p���+����CV�                                    Bxn��  �          @��
�Å�xQ쿓33���CTk��Å�X���p�����CQ�                                    Bxn��  �          @�33��G���
=������HCb=q��G��y����z���p�C`�                                    Bxn�0  �          @�(���\����>�@n{C�׿�\���У��NffC��
                                    Bxn��  �          @�R�\)��=q>�ff@c�
C{���\)��p����
�D��C{h�                                    Bxn�|  �          @�p��
=��Q�=#�
>���Cz���
=��\)��z��w
=Cy��                                    Bxn�"  �          @��   ��{���s33Cy���   ���
��\���HCx�{                                    Bxn��  
�          @�
=������zᾣ�
�5C}�=������G�����33C|�
                                    Bxnn  �          @�  �]p���Q�>.{?�
=Co��]p���녿�G��O33Cn#�                                    Bxn  �          @�{�Z�H����>#�
?�  Ck���Z�H��(�����F�\Cj�q                                    Bxn!�  �          @��i���E��B�\��
C\8R�i���.�R��\)��ffCX�
                                    Bxn0`  �          @��H�~{�aG�>#�
?�
=C]���~{�Z=q�^�R��C\��                                    Bxn?  �          @��\���\�hQ�?O\)A=qC]�H���\�l�;�z��=p�C^:�                                    BxnM�  �          @����B�\��G�?0��@�{Cl0��B�\��G��(����Cl5�                                    Bxn\R  �          @��R��H��Q�?p��A\)Cs\��H���\���H��
=CsaH                                    Bxnj�  �          @�33�o\)���\?��@��RCi���o\)���׿u�  CiQ�                                    Bxny�  �          @�z��h����=q>�@�
=Cg}q�h����  �fff��Cg{                                    Bxn�D  �          @�ff�
=��z�>�z�@5�CvW
�
=��\)��ff�L��Cu��                                    Bxn��  �          @�
=��
��
=>�G�@�\)Cxn��
�����{�6�RCx
=                                    Bxn��  �          @���<��?�������\C*{�<��?Tz�n{���C$O\                                    Bxn�6  �          @�\)�~{@xQ쿦ff�R=qC��~{@��H��Q�p��C+�                                    Bxn��  �          @�\)���@�{����b�HC�3���@�
=�8Q���C{                                    Bxnт  �          @�p���z�@���Q��_�
CG���z�@�{���
�(��C �)                                    Bxn�(  �          @�����
@�����-�Cc����
@��H>�  ?�{CxR                                    Bxn��  �          @�����(�@��Ϳ޸R�W�C	����(�@�ff�W
=�˅C��                                    Bxn�t  �          @�G��θR@|(����G\)C�H�θR@�Q��
=�HQ�C�)                                    Bxn  �          @�����(�@XQ쿝p����C�H��(�@e�.{��p�CxR                                    Bxn�  �          @�  ��33@2�\��33���C����33@@�׾����
=C
=                                    Bxn)f  �          @��
��{@�\)���R�z�C
Ǯ��{@�33?���AC��                                    Bxn8  �          @�33��(�@���>��?�B�Ǯ��(�@��R?���Aw�B��q                                    BxnF�  �          @�z���33@���#�
����B�Ǯ��33@���?�p�AV=qC&f                                    BxnUX  �          @�  �e@�\)�����$  B�#��e@ڏ\?5@�  B�k�                                    Bxnc�  �          @����i��@�
=������B�
=�i��@�=q?=p�@�\)B�aH                                    Bxnr�  �          @��a�@�Q�=�Q�?(��B�(��a�@�p�@Q�A�p�Bힸ                                    Bxn�J  �          @����}p�@�{���w�B�G��}p�@���?\A4(�B�u�                                    Bxn��  �          @����(�@�33?aG�@�{C����(�@��
@�RA��C�q                                    Bxn��  �          @�\)�[�@����'
=��{B�k��[�@�z�
=q����B���                                    Bxn�<  �          @����z�H@����
��Q�B��z�H@ָR��\)�ffB�                                    Bxn��  T          @���l��@�  �����
=B�Q��l��@ۅ���Y��B�3                                    Bxnʈ  
�          @����r�\@����  ����B�Q��r�\@��þB�\��z�B�u�                                    Bxn�.  T          @�
=��@�����  ��Bۅ��@��H� ���z{B�p�                                    Bxn��  �          @�z��	��@��������
=BܸR�	��@�  �33��
=B�\)                                    Bxn�z  �          @���,(�@�  �����\)B�Q��,(�@޸R�G��r�\B�Q�                                    Bxn	   T          @�
=�@  @Ǯ�z�H��ffB�\)�@  @�z�����S\)B�\                                    Bxn	�  T          A33�_\)@�(��R�\���B�z��_\)@�=q�����(�B��                                    Bxn	"l  �          A=q�K�@���e���\)B��K�@�33�����B�Ǯ                                    Bxn	1  �          A Q��QG�@љ��QG���ffB���QG�@��������B��                                    Bxn	?�  �          @���XQ�@љ��Dz�����BꙚ�XQ�@��c�
��(�B�p�                                    Bxn	N^  �          A z��Z=q@���7
=��G�B�33�Z=q@�
=�&ff��B�\                                    Bxn	]  �          @�{�XQ�@�(��S33����B��)�XQ�@��H��z���\B�                                      Bxn	k�  �          Aff�_\)@�(�������{B�8R�_\)@��H��
=�[33B�ff                                    Bxn	zP  �          @�(�����@��H��(��i�B�k�����@�z�=���?E�B�
=                                    Bxn	��  S          @�����p�@��Ϳ��H�f�\B�.��p�@�{>�?z�HB��
                                    Bxn	��  �          A (���\)@��p���  B����\)@�G���Q�(�B�Ǯ                                    Bxn	�B  �          A Q�����@�p�����w33B�������@�Q켣�
��B��                                    Bxn	��  �          A ������@�p���=q�S�B�z�����@��>�33@!�B�3                                    Bxn	Î  �          A z��{�@�Q���
�M��B�L��{�@�\)>�(�@FffB�R                                    Bxn	�4  �          A�R�y��@�G���  ��B����y��@�\?��
@���B��q                                    Bxn	��  �          A����{@��ÿ�z��z�B�\��{@�?aG�@\B���                                    Bxn	�  T          A(���(�@���p��A��B� ��(�@�(�?��@u�B�#�                                    Bxn	�&  �          A�R���@����
=q�v{B�����@߮=u>��B�\                                    Bxn
�  �          @���hQ�@��H�*�H��z�B��q�hQ�@�33��G��L(�B�33                                    Bxn
r  �          @�ff�j=q@�ff�7
=��ffB���j=q@��ÿ(����\)B�                                      Bxn
*  �          A (��G�@��C33���B�\�G�@陚�E���Q�B�B�                                    Bxn
8�  �          A��'
=@�\)�@�����B����'
=@�\�!G����RB���                                    Bxn
Gd  �          @�z��ff@�{����^{Bˀ ��ff@���?��@�z�B��H                                    Bxn
V
  �          @�p��J=q@�����z�B���J=q@�33>�33@!G�B��=                                    Bxn
d�  �          A z�
=q@�  ��\�k33B��ÿ
=q@��?\)@~{B��R                                    Bxn
sV  �          A=q��Q�@����R�}�Bŀ ��Q�A ��>\@*�HB��
                                    Bxn
��  �          A��z�@������p�B�
=�z�@��=�G�?G�B�p�                                    Bxn
��  T          AQ��@  @�Q��L(����B�W
�@  @���B�\��  B�                                    Bxn
�H  T          A=q�C�
@�=q�S�
��33B�B��C�
@�׿n{����B�33                                    Bxn
��  �          A ���#33@���R��=qB�#��#33@��H���
�#�
B�#�                                    Bxn
��  �          A\)�!�@�\�*=q��\)B���!�@����\)�uB���                                    Bxn
�:  �          @���"�\@��H�R�\��{B�Ǯ�"�\@�G��c�
���
B�G�                                    Bxn
��  �          @�{���@��
�b�\��\)BҸR���@��Ϳ�{� z�BϸR                                    Bxn
�  �          @����I��@���j�H���B���I��@����R�1��B��                                    Bxn
�,  �          @�G��p��@����g
=�܏\B��f�p��@���\�3\)B��H                                    Bxn�  �          @����r�\@��H�^�R�ԣ�B����r�\@�������#\)B�Q�                                    Bxnx  �          @��H�tz�@�Q��b�\����B�z��tz�@��
����;�B��                                    Bxn#  �          @��^{@���G
=��=qB�=q�^{@љ���G���Q�B�                                    Bxn1�  �          @��5�@����=p�����B�Q��5�@љ��\(���33B���                                    Bxn@j  �          @�z��a�@�������\B�\�a�@�{��Q�0��B�p�                                    BxnO  T          @�R���@�\)�Tz����
C�=���@�=q��z��U�C��                                    Bxn]�  �          @����  @��
�Mp��ҏ\C����  @�ff��z��T��CJ=                                    Bxnl\  �          @�z���33@4z��p�����HC33��33@z=q�'
=��(�C�)                                    Bxn{  �          @�\)��(�?�\���H�
C"���(�@A��QG���z�C�R                                    Bxn��  �          @����{@s33��{��CaH��{@�\)�,(����C�q                                    Bxn�N  �          @�\����@g
=���\��Cn����@��H�8Q����RC+�                                    Bxn��  �          @����Q�@i���s�
���\C���Q�@��R����=qC�                                    Bxn��  �          @�33��Q�@2�\�|(���C+���Q�@|(��1���ffCh�                                    Bxn�@  �          @����  @���{�����C:���  @dz��9�����\C5�                                    Bxn��  �          @���z�@1��{����C��z�@|(��0����
=C
                                    Bxn�  �          @������H=�\)���R�p�C3Y����H?Ǯ���
�\C%                                    Bxn�2  �          @�G�����=u����;�HC3T{����?�\��G��-�C�
                                    Bxn��  �          @�z����R��������J��CB�����R?B�\����M�RC)                                    Bxn~  T          @޸R���\�Ǯ��\)�733CF޸���\>W
=���R�B�C1^�                                    Bxn$  �          @޸R����ff��(��@{CO�����u��G��Uz�C7T{                                    Bxn*�  �          @أ����R��������.�RCJ� ���R�.{��
=�@�C68R                                    Bxn9p  �          @��a�?�
=���
�W�
C���a�@Y����\)�)�RC\                                    BxnH  �          @����  @e��_\)��=qCk���  @�����
��\)C	�3                                    BxnV�  �          @��H��  @����z����C�{��  @�녾�=q��CE                                    Bxneb  �          @�G���z�@�
=�N�R��
=C���z�@�녿�G��AG�C+�                                    Bxnt  �          @�G���(�@��R�A����HCz���(�@�
=��p���
CE                                    Bxn��  �          @��H��\)@b�\�j�H��G�C����\)@��\��R��Q�C
�                                     Bxn�T  �          @陚�ƸR@,���,������Cz��ƸR@^{�����K33Cٚ                                    Bxn��  �          @�  �أ׿����B�\���C@!H�أ׿�ff�.{��33C>�H                                    Bxn��  �          @�(���  �O\)?
=q@�Q�C:����  �p��>k�?��C;�3                                    Bxn�F  T          @�  �ʏ\>�G������"{C0\�ʏ\?L�Ϳh���=qC,�=                                    Bxn��  
�          @�\)�����녾8Q��  C=�������G��\)����C<��                                    Bxnڒ  �          @������H����?z�@��HC>�\���H��Q�=�Q�?.{C?}q                                    Bxn�8  �          @�����ͿY��?�z�A
=C:���Ϳ���?L��@���C=��                                    Bxn��  �          @���߮�:�H?���AffC9�3�߮����?^�R@�\)C<�                                    Bxn�  �          @���\)��@
�HA��RC8����\)��G�?���Aq�C>�{                                    Bxn*  �          @�=q��\=u?�
=AV{C3����\�
=q?���AJ�RC8T{                                    Bxn#�  �          @�z���ff�.{?�A9�C5p���ff�(��?��\A%G�C9p�                                    Bxn2v  �          @�����
�O\)?���A-G�C:�R���
��(�?u@���C>
                                    BxnA  �          @���녾\)?���A�RC5.��녿�?�G�AffC8p�                                    BxnO�  �          @��
�������\(���CI)���Ϳ޸R�����n{CE)                                    Bxn^h  �          @�����   �\�`��CF�R��녿޸R����=qCD�H                                    Bxnm  �          @�ff��33���H�c�
�Q�CC�R��33��ff�����h(�C?��                                    Bxn{�  
(          @�{���R��G���ff�%��C8s3���R�L�Ϳ���3�
C4u�                                    Bxn�Z  �          @�p��������E���CH����Ϳ�z����PQ�CD�)                                    Bxn�   �          @ۅ��z��|(��Y����\)CX+���z��XQ��(���  CT�                                    Bxn��  �          @����{���!G���z�C[+���{�z�H�
�H��G�CW                                    Bxn�L  �          @�p���(����ÿ5��p�C`\)��(����R�
=��
=C\                                    Bxn��  �          @�=q��
=��
==�\)?
=qC^(���
=��{��\)�U��C\aH                                    BxnӘ  �          @�
=���H��  �����Q�Ci  ���H��Q����\)Cf��                                    Bxn�>  �          @��\)��p����ͿQ�Cl
=�\)����{���RCj�                                    Bxn��  �          @����������?�=qA
=Ca����������O\)��G�Cb.                                    Bxn��  �          @����\)��\)?��HA?�C_����\)��{��{�1�C`Ǯ                                    Bxn0  �          @�33�������
?��AH��C]�H��������k���=qC_Y�                                    Bxn�  �          @����{��  ?u@��RC[����{��G��:�H����C[�)                                    Bxn+|  �          @�G���p����H?�33A�C]�{��p���{�z���Q�C^=q                                    Bxn:"  T          @������
���?�G�Az�C^W
���
���R�@  ����C^�H                                    BxnH�  T          @�G���
=��\)?��RAC33C\�f��
=��
=�u����C^�                                    BxnWn  �          @�\�����w
=?���A(�CU�H�����~�R�����P��CVz�                                    Bxnf  �          @�G���G��W
=?�R@�
=CS@ ��G��W
=�&ff��ffCS8R                                    Bxnt�  �          @�����ff��
��p��R�\CI5���ff�G���
=�*�RCF�R                                    Bxn�`  �          @��H��G��=p����mG�C:5���G���������ffC4:�                                    Bxn�  �          @��H�׮���R�
=����C6�H�׮>��z���ffC/�f                                    Bxn��  �          @�Q���ff>���(���
=C2����ff?W
=����l��C,ٚ                                    Bxn�R  �          @�z�����?�G��#33���C*�f����?�\)� ����=qC#aH                                    Bxn��  �          @�\��p�?�\�7���z�C/}q��p�?��
�\)��33C&��                                    Bxn̞  �          @�\)��  ���=q���RC5
��  ?E���\��  C-5�                                    Bxn�D  �          @ڏ\���
�����1G���ffC=�3���
<��>{��G�C3                                    Bxn��  �          @�{��33>�z��aG��z�C0k���33?����J=q�=qC!(�                                    Bxn��  �          @�
=�XQ�?��
����L�C+��XQ�@Z=q�o\)��C�R                                    Bxn6  �          @�\)���R?�z�����}
=C33���R@s�
��(��6�B�=q                                    Bxn�  �          @����\)@   ��=q�~�RC
W
�\)@����z��9
=B�B�                                    Bxn$�  �          @����S33�u��=q�i=qCDB��S33?�
=�����fp�C O\                                    Bxn3(  �          @��H�e�z����
�_ffC=(��e?��H��ff�T��C��                                    BxnA�  �          @�{�u��  ���H�B��CF��u?\)��\)�JffC+�                                     BxnPt  �          @�\)���\������z��7�RCCp����\?!G���  �=ffC+�q                                    Bxn_  �          @ٙ����H��\��G��G�CML����H�����33�-z�C;)                                    Bxnm�  �          @ָR��{�aG����\�;�\C?5���{?�  ����:\)C'J=                                    Bxn|f  �          @��H��
=�����Q��G�C7�\��
=?������9\)C�3                                    Bxn�  �          @θR��ff��
=�U�����C8aH��ff?^�R�O\)��G�C*�3                                    Bxn��  �          @����H��Q����\�4�CFL����H>��������>�\C.�                                     Bxn�X  �          @ҏ\�e����(��`33C<^��e?�ff���T  C�f                                    Bxn��  �          @׮���
?(����\)(�C�����
@>�R�����gp�B��                                    BxnŤ  �          @љ���(�?J=q�����C�{��(�@A������fG�B��                                    Bxn�J  �          @�(���p�@�R��z�\B��
��p�@�{�����/=qB�u�                                    Bxn��  �          @�=q?�{@G������~�B=G�?�{@~{���H�3  B���                                    Bxn�  �          @�  ?��H@\)��G�{BR�?��H@�����  �1(�B�k�                                    Bxn <  �          @�\)?�z�@-p������vQ�By��?�z�@�p����"��B�Q�                                    Bxn�  �          @ٙ�?��@u����R�K�HB���?��@����Vff�B�{                                    Bxn�  �          @˅��
=@���@  ���B�W
��
=@�녿���K
=B��                                     Bxn,.  �          @�Q��qG�@�p���{�pz�C ��qG�@�>��R@4z�B���                                    Bxn:�  �          @�33�h��@���9���ٮC:��h��@��H���\�  B�.                                    BxnIz  �          @�=q�AG�@B�\���R�8��C�{�AG�@����<����B�z�                                    BxnX   �          @�ff�g
=?�������Kz�C�=�g
=@Dz��vff�Q�C�)                                    Bxnf�  �          @˅�vff?8Q���(��M{C)^��vff@#33��
=�)�RCn                                    Bxnul  �          @�(���z�?c�
����HffC'����z�@1���=q�#�RC)                                    Bxn�  �          @�Q���{?�
=����)��C @ ��{@Tz��b�\��33C�R                                    Bxn��  �          @����  >�Q���p��O
=C/8R��  @���33�2�HC8R                                    Bxn�^  �          @�\��>��������*Q�C/�{��@�R������C��                                    Bxn�  �          @��H��=q�.{�E��G�C:)��=q?z��G
=�У�C.�q                                    Bxn��  �          @�\)���H���Ϳ�=q�G�CDG����H���
��(��n�\C?k�                                    Bxn�P  �          @�=q���H��\�5��  CH�����H��=q��ff�[\)CD��                                    Bxn��  �          @�33��ff�Q�>8Q�?˅CQ����ff�E��\)�Q�CPn                                    Bxn�  �          @�ff�����dz�?L��@���CV�����fff�+���ffCV8R                                    Bxn�B  �          @�{����j=q?��A@��CW�{����x�þL�Ϳ���CYE                                    Bxn�  �          @ə���Q��l(�?�Aw�CY� ��Q�����=��
?:�HC\Y�                                    Bxn�  T          @�z�����(Q�Ǯ�g
=CMO\�����׿���Q�CJ#�                                    Bxn%4  �          @�  ��=q�녾�33�FffCH�
��=q���H��(��-CE�)                                    Bxn3�  �          @Ϯ��33�ff�����CG���33��  ��  �2�RCD�                                    BxnB�  �          @љ���G���Q�=�\)?��CC���G���=q�(����CB�                                    BxnQ&  �          @Ϯ��Q쿯\)� ������C@ٚ��Q��������C8^�                                    Bxn_�  �          @�=q���Ϳ�z�� ����{CAff���;�z��7
=����C6�\                                    Bxnnr  �          @�  ��
=��
=��G��(�CC� ��
=��녿����k�C>��                                    Bxn}  T          @Ǯ���׿�=q�G���
=C?�\����>���S33��C133                                    Bxn��  �          @�ff����E����7(�C>Q����?�����
�3��C%�                                    Bxn�d  �          @����|�;����=q�I�
C:�R�|��?�  ��33�=�RC+�                                    Bxn�
  �          @����G���G�>.{?�(�C5!H��G��.{=�G�?�\)C5��                                    Bxn��  �          @�33���\>��5��  C/
=���\?�����H��=qC#��                                    Bxn�V  �          @�������@Q��z���Q�C(�����@'��+����C:�                                    Bxn��  �          @�����z�?c�
���H��G�C+���z�?�  ���H�:�HC%)                                    Bxn�  �          @�  ��z�^�R��\)�aC=.��z�u��{���RC6�{                                    Bxn�H  �          @��������������\CL�������녿�z��_�CI�                                    Bxn �  �          @�(����H�G���\)�0��CJ!H���H�33�}p���\CH!H                                    Bxn�  �          @�\)���E�@A�ffC;�q����ff?���As33CCL�                                    Bxn:  �          @�  �s�
��Q쿎{�n�\CK���s�
���Ϳٙ����
CD�                                    Bxn,�  �          @��R����@�
�#33��33C5�����@I������Z=qC�H                                    Bxn;�  T          @�������@C33� ������C�����@fff�
=����C\                                    BxnJ,  �          @�(���  ?�z��%�܏\C#� ��  @�����33C�                                    BxnX�  �          @��R��=q@
=q�'
=��z�C����=q@B�\��  �tQ�Cc�                                    Bxngx  �          @�\)����?��HQ����C�{����@A��z�����C��                                    Bxnv  �          @�=q�\)@Tz��&ff��33CB��\)@��H��G�� ��CB�                                    Bxn��  �          @�
=���\@H���)�����C���\@|�Ϳ����.=qC	�{                                    Bxn�j  �          @�  ��
=@��5���(�C����
=@R�\��\)�{�C�                                    Bxn�  �          @\���?�(��Z=q��Cٚ���@L����\���C@                                     Bxn��  T          @�{��p�?�\�S33� �HC 0���p�@>{�G���ffC�
                                    Bxn�\  �          @ə���=q?���^�R���C(u���=q@Q��/\)��z�C�
                                    Bxn�  �          @������?��\�W�����C&������@#33�"�\���HCY�                                    Bxnܨ  �          @�33��z�W
=�b�\�G�C6B���z�?��
�S�
��33C&��                                    Bxn�N  �          @ҏ\���\�
�H�G���(�CJ0����\�.{�o\)���C;B�                                    Bxn��  �          @��H���H��
�,����{CI
=���H�@  �S�
��Q�C;�                                    Bxn�  �          @�  ��=q� ���ff���HCME��=q��z��<����CB�
                                    Bxn@  �          @�����p���=q�\)����C@�q��p��#�
�\)��z�C4�                                    Bxn%�  �          @ٙ���{@�z��n{�\)B��H��{@����\�
�RB���                                    Bxn4�  �          @��H����@�ff��33��RB��ÿ���@ƸR��33�i��B�{                                    BxnC2  �          @Ӆ��{@�
=�{��
=B�\��{@����p��O�B՞�                                    BxnQ�  
�          @�  �h��@�z���ff�E�\B̮�h��@��H�0  ��Bţ�                                    Bxn`~  �          @љ��h��@�������Hz�B���h��@��H�5��\)Bų3                                    Bxno$  �          @��H>Ǯ@�
=���H���B�k�>Ǯ@�녿޸R�l��B�(�                                    Bxn}�  �          @߮>�ff@��
���R��B�ff>�ff@�  ���
�l(�B�\)                                    Bxn�p  �          @�p��5@*�H�����qB���5@��R���\�#\)B�L�                                    Bxn�  �          @�33?��
@|(�����ZffB���?��
@�  �a����HB�p�                                    Bxn��  �          @�?�33@��
��Q��I
=B��=?�33@�G��I����(�B���                                    Bxn�b  �          @�z�?���@~{��
=�V�B��R?���@����`  ��Q�B�=q                                    Bxn�  �          @ᙚ?��@��
����I=qB�B�?��@�  �B�\��
=B�#�                                    Bxnծ  �          @��@�
?���ҏ\aHB\)@�
@�ff�����Bz�B��                                    Bxn�T  �          @��@(Q�?���(�L�B��@(Q�@����
=�4(�Bn�                                    Bxn��  �          @�(�@S33?˅��z��p��A͙�@S33@����33�/G�BNQ�                                    Bxn�  
�          @�(�@tz�@���
=�X\)A�(�@tz�@�z��������BC�                                    BxnF  �          @�33��
=@HQ���Q��j�HB�=q��
=@�{�h���	p�B͙�                                    Bxn�  �          @ƸR��33@G
=���\�](�B��H��33@���P����z�Bؙ�                                    Bxn-�  �          @�=q��\)@Vff�����[�RB�k���\)@���N{��B�G�                                    Bxn<8  �          @��?�Q�@QG����
�f  B�z�?�Q�@��
�j=q�z�B��                                    BxnJ�  �          @�G���@�
=���8��B�p���@�
=����Q�B��3                                    BxnY�  �          @�G�?�(�@�����Q��:B�ff?�(�@�33�*�H��B���                                    Bxnh*  �          @�  @
�H@�����  �>�B�@
�H@�\)�@  ��B�                                      Bxnv�  �          @��?��@~{�Å�^z�B�u�?��@�
=�r�\���\B�8R                                    Bxn�v  �          @���z�@�����  �`�B����z�@�\)�j=q��\)B�B�                                    Bxn�  �          @�\)>��H@�����p��Q��B�>��H@����XQ����HB���                                    Bxn��  �          @�=q?���@�33����R(�B�{?���@У��`  ��p�B��H                                    Bxn�h  �          @���?�z�@y����ff�^�RB���?�z�@�ff�x������B��                                    Bxn�  �          @�@%�@8���ə��i�\B@�H@%�@��H����z�B�k�                                    Bxnδ  T          @�p�@Q�@�ȣ��h�HB{@Q�@��
��\)���BZz�                                    Bxn�Z  �          @��@vff?}p�����g(�Af�\@vff@h������2B-�R                                    Bxn�   �          @��@R�\?�p���=q�oA��@R�\@p����p��2Q�BC�                                    Bxn��  �          A{��@����(��]B��ͼ�@�  �}p���B��                                    Bxn	L  �          A�>\@�33���
�V�B�  >\@�Q��u���=qB���                                    Bxn�  �          Az�>Ǯ@����{�W��B���>Ǯ@���xQ���\)B��H                                    Bxn&�  �          A�
��  @�{���H�TQ�B��;�  @��H�o\)��(�B��f                                    Bxn5>  �          A z��G�@�{���H�PG�B�z��G�@�
=�aG��ϮB��3                                    BxnC�  �          A ��>���@�ff��  �h(�B�>���@�  ���R��\)B��)                                    BxnR�  �          @�녾�\)@j=q��{�m=qB��q��\)@����|(����B�Q�                                    Bxna0  �          @����Ǯ@tz�����p  B��q�Ǯ@��
�����\)B�\                                    Bxno�  �          @�33�n{@�ff��(��6�B�z�n{@��
�(Q���(�B�u�                                    Bxn~|  T          A   �s33@�(����H�;�\B���s33@�p��6ff���HB¨�                                    Bxn�"  �          @�ff���
@�z���G��:p�Bʙ����
@���,(�����BĊ=                                    Bxn��  �          @�R����@�{���/
=BԽq����@��
�{��33B̽q                                    Bxn�n  �          @�(���p�@�  ��G��5�RB�ff��p�@���'����B�G�                                    Bxn�  �          @����R@�(�����%��B�z���R@�����\����B�=q                                    BxnǺ  �          A����@�  ��z��:z�B�.���@�33�B�\��=qB�                                    Bxn�`  �          A (�����@�p��ʏ\�RBиR����@�  �dz��ՅB�\)                                    Bxn�  �          A33�8Q�@������`ffBƔ{�8Q�@�p��~�R���
B��f                                    Bxn�  �          Ap���G�@n{��{�}��B�G���G�@������Q�B��
                                    BxnR  �          Aff��33@�G���ff�k\)BӸR��33@�\)��z���B�\)                                    Bxn�  �          A (��8Q�@g
=���|(�B����8Q�@���\)�p�B�.                                    Bxn�  �          @��R��=q@=p����\)B�uþ�=q@�=q��
=�"  B�Q�                                    Bxn.D  �          @�{�8Q�@1���\B��8Q�@����H�&�B�aH                                    Bxn<�  �          @����@z���
=�~\)C�3���@�33��ff�%(�B�\)                                    BxnK�  T          @�G��r�\@�R��=q�X33C���r�\@�\)����{C                                     BxnZ6  �          @�׿�(�@33��33�
C޸��(�@����133Bߨ�                                    Bxnh�  �          @��H�&ff?�ff��\�qB�ff�&ff@���
=�D�B��                                    Bxnw�  �          @�p��fff?�G����\B���fff@�����DBȳ3                                    Bxn�(  �          @��H�:�H?W
=��  £#�C
�:�H@����ə��`�B���                                    Bxn��  �          @�����
>�\)��\)¯�BԀ ���
@p����Q��s�\B�z�                                    Bxn�t  �          @��׾.{���
��  ®�{Cq녾.{@O\)�ᙚ�{B��                                    Bxn�  �          @�
=�#�
�#�
��ff°��Ca�;#�
@W���#�B�Q�                                    Bxn��  �          @�
=�������
���R¯u�CBG�����@\����z��}Q�B���                                    Bxn�f  �          @��R��
=������p�«��CZO\��
=@Mp���\)=qB�                                      Bxn�  h          @�z�G���{��33Cp)�G�@(���\)Q�B�33                                    Bxn�  �          @�  �E��J=q����£�\CaǮ�E�@333����)B�Ǯ                                    Bxn�X  �          @����\�!G����¨#�Cf�{��\@A���\)  B�Q�                                    Bxn	�  �          @�{��
=�L����\¦z�CrL;�
=@1G���33�fB�#�                                    Bxn�  �          @�p��p�׿�ff��{�3Cn�p��@33��=qǮB�(�                                    Bxn'J  �          @�G��������33ǮCss3���?�{��\��C�                                    Bxn5�  �          @�zῦff�-p���R�3Ctff��ff?h�������C޸                                    BxnD�  �          @���?5������Tz�C�Q�?5���R��  �C���                                    BxnS<  T          @���?.{�����\�N{C��?.{��
=��33aHC��                                     Bxna�  T          @����:�H�����C�����?(����\©�HB�Q�                                    Bxnp�  �          @�p��5������Cy�H�5?�z����HB�B�aH                                    Bxn.  �          @�  ��Q��  ����\CrͿ�Q�?������\C��                                    Bxn��  �          @��H��\)�6ff���
=C�9���\)?Y����Q�¦��B��                                    Bxn�z  T          A Q�\)�|����ff�pz�C��{�\)��=q���«\CM�                                    Bxn�   �          A �ÿ:�H�����=q�O�C��=�:�H��33��(�W
Cr��                                    Bxn��  �          @���W
=��\)��33�@�C��)�W
=��33���qC��H                                    Bxn�l  �          @���@Vff��ff�5����C���@Vff��z���Q��3�RC��                                    Bxn�  �          @���@"�\�޸R�Q���ffC��@"�\��G���33�-�HC�aH                                    Bxn�  �          @��@A���z῝p��  C���@A����������	z�C�]q                                    Bxn�^  �          @�{@%���(��u��ffC�J=@%���(���=q�33C�S3                                    Bxn  �          @�{@,(���(���(��vffC��@,(�����33�#(�C�K�                                    Bxn�  �          @�@�
��(��E���(�C��R@�
�������33C��                                     Bxn P  �          @����
=��  ����4�RC~\��
=���H��
=��Ci޸                                    Bxn.�  �          @��������(����&��C�Ῠ������(�.Cqz�                                    Bxn=�  �          @��Ǯ���������
C�Ǯ�Fff��z��y��Cs=q                                    BxnLB  �          @�R�����
��{�	�C|�)���L����
=�q{Cp#�                                    BxnZ�  T          @�\)�
=q��33�p����
=Cz� �
=q�g
=��{�_=qCo{                                    Bxni�  �          @����)���\�9������Cv}q�)���\)���B\)Clu�                                    Bxnx4  �          @���R��z��   ���Cx���R�������(�Cr�                                    Bxn��  �          @�33��\��p��(������Cy����\������>�CqW
                                    Bxn��  �          @�
=��\)���������=qC�Ϳ�\)���
��ff�:�
Cz�R                                    Bxn�&  T          @���(������r�\����C~)��(��aG���ff�f�Cs�3                                    Bxn��  �          @����(��Ӆ�	�����Cy� �(����������,G�Cs�                                    Bxn�r  �          @�{��
�ָR�(�����\C{���
���H�q����Cw�=                                    Bxn�  �          @ᙚ�У���{��Q��=C�"��У���{��ff�(�C|��                                    Bxn޾  �          @ۅ����ff��G��*{C~W
����=q���=qCzaH                                    Bxn�d  �          @׮�	����=q>\@_\)Cy�3�	�������(����Cw�H                                    Bxn�
  �          @�z��G
=��{?fff@�p�Cq\)�G
=�������RCp8R                                    Bxn
�  �          @�
=�<�����ÿ��Z�RCo�H�<���o\)�i����HCg                                    BxnV  
�          @Ӆ�X����G��z=q�(�Ce�q�X�ÿ�
=���
�[=qCNQ�                                    Bxn'�  �          @ڏ\�8�������s�
�	  Cn�\�8������(��a�C[\)                                    Bxn6�  �          @�\�<(����\�y���z�Co��<(��%����a  C]W
                                    BxnEH  �          @��H��������33�&��Cy�{����
=q�ָRǮCd�R                                    BxnS�  �          @��5��Q������ffC�Y��5�1�����C�                                     Bxnb�  �          @�{�Dz���
=�.{��\)Ct���Dz����H�P������Cq=q                                    Bxnq:  �          @޸R�W
=��=q���R�#�
Cq
=�W
=��p��L����33Cl�q                                    Bxn�  �          @ڏ\�,�����H�O\)��{Cv��,����{�h�����CqaH                                    Bxn��  �          @�z��K���ff�\�L(�Cqٚ�K������L����(�Cm��                                    Bxn�,  �          @�(���H�ȣ�?�Q�A Q�Cx���H��녿��R����CxB�                                    Bxn��  �          @�(�� ���ʏ\?=p�@ƸRCxaH� �������=q����Cv��                                    Bxn�x  �          @��
�1G���\)������
Cv�1G���{�`  ��  Cq�H                                    Bxn�  �          @�=q�	����=�?�  C{}q�	�����@������Cy@                                     Bxn��  �          @�(���G���?��@��C�����G��Å�0����G�C��                                    Bxn�j  �          @׮�8Q���G�@
=A���C��
�8Q��У׿��H�&�RC���                                    Bxn�  �          @ָR��{��(�@�RA�ffC��q��{��{���
�ffC��                                    Bxn�  �          @�p���Q����>��@X��C���Q������8Q���(�C~Q�                                    Bxn\  �          @ڏ\�	����p�>.{?���C{���	����{�>�R�Џ\CyT{                                    Bxn!  �          @�G���G���  @,��A��\Cٚ��G���녿���=qC��H                                    Bxn/�  �          @��Ϳ�����\�H����33C�޸����c�
���H�a�C}}q                                    Bxn>N  �          @��Ϳp����(��XQ���z�C�uÿp���^�R��=q�h�
C~�f                                    BxnL�  �          @�33��  ��
=�:=q��G�C�>���  �r�\���R�YQ�C5�                                    Bxn[�  �          @��>.{��  ������{C�Ǯ>.{��������6(�C��                                    Bxnj@  �          @�=q?��ָR�Q���
=C�l�?���ff��Q���C���                                    Bxnx�  �          @ٙ��u���ͿTz����C��
�u��ff�n�R�{C��                                    Bxn��  �          @�33���������\)��C�箿��� ���ȣ�� Cu��                                    Bxn�2  T          @�Q쿠  ��z��x���Q�C�uÿ�  �2�\�����|��Cu�=                                    Bxn��  �          @�  ��{������\��C{aH��{�l(����R�C(�Cs0�                                    Bxn�~  �          @�����z���\)?fffA{C_���z���(���ff�B�HC_8R                                    Bxn�$  �          @�\)�0  ���H@\)A�ffCq�{�0  �����{�E�Ct
=                                    Bxn��  �          @�{�*�H��p�@ ��A��RCr���*�H�����Q��S33Cu
=                                    Bxn�p  �          @θR��G���
=�W
=�
{C����G����H�\(��p�C�s3                                    Bxn�  �          @�{��{���ÿ��f�HC�C׾�{��33�~{�/�
C���                                    Bxn��  �          @��H�E����ÿ˅�~�\C���E��������
�4�C��3                                    Bxnb  �          @���W
=?�����Q��)B�{�W
=@k���  �A�B�                                    Bxn  �          @����G��!G����B�CE.�G�@
=�����pQ�C�H                                    Bxn(�  �          @��Ϳ�
=���H��.Ci�\��
=?�\)���HW
CW
                                    Bxn7T  �          @ҏ\�&ff�(Q�����]�RCaY��&ff>\���Ru�C+�R                                    BxnE�  �          @�\)�U��L����ff�0  C_ٚ�U��\)��ff�i
=C=�
                                    BxnT�  �          @�G��"�\�����
=�d�C\�{�"�\?0�����
ǮC$ٚ                                    BxncF  �          @�녿�\�  ���R�yG�Cg�H��\?B�\�����C�3                                    Bxnq�  �          @ָR�J=q�\)��G�� �Cg�
�J=q�������R�l
=CJ�3                                    Bxn��  �          @�  �i�����\����Q�Cf��i������ff�K��CR#�                                    Bxn�8  �          @�(������\)��\)�eCd#�����R�\�qG��\)CY�\                                    Bxn��  �          @�  �������׿��H�X��C]�f�����0���QG����CS^�                                    Bxn��  �          @�����
=���\?��
AG�C@���
=��\)>��@Q�CC&f                                    Bxn�*  �          @�����
�z�?�33A�
=CHB����
�0  ?(�@��\CN�                                    Bxn��  �          @����p��{@(�A�=qCL�=��p��W
=?^�R@��CS��                                    Bxn�v  �          @�(���{��@/\)A��RCG���{�^{?�z�A  COp�                                    Bxn�  �          @�\���ÿ���@E�A��RC>������%�@�
A~{CH�{                                    Bxn��  �          @�\��=q>u@E�Aȏ\C1�3��=q���@1�A�{C?��                                    Bxnh  �          @�=q��(�?B�\@{�B��C,����(����@q�A���C@
                                    Bxn  
�          @�Q���p�?��@��BQ�C ����p��#�
@�p�B'�\C;)                                    Bxn!�  �          @����Q�?��@�{B$Cn��Q�:�H@�  B2ffC<J=                                    Bxn0Z  �          @޸R���\@\)@�\)B-�HC
���\�k�@��HBK�
C7\                                    Bxn?   �          @��
��Q�@!G�@�(�B-z�C�
��Q쾏\)@��BJ  C7��                                    BxnM�  �          @�R����@/\)@�B8��C�{���;�  @�33BZffC7aH                                    Bxn\L  �          @�Q��j�H@E�@�33B@��C
=�j�H�#�
@�p�Bm  C4��                                    Bxnj�  �          @ָR?G�@�p�@W�A���B��=?G�@>{@�z�Bu��B�z�                                    Bxny�  �          @�p�?�@���@�Q�B
z�B���?�@<(�@��Bz\)Bi(�                                    Bxn�>  �          @�@z�@�Q�@�ffB  B�@z�@3�
@�33B\)BVff                                    Bxn��  �          @�ff?�p�@Å@�ffB�
B�(�?�p�@H��@�
=BtQ�BgQ�                                    Bxn��  �          @�R?��@�{@�{B%
=B�{?��@ff@�p�B�z�Bi��                                    Bxn�0  �          @�(�>aG�@���@�=qBT�B�Q�>aG�?Tz�@��B�  B���                                    Bxn��  �          @�녾�z�@���@�\)B=\)B�aH��z�?��@�B�aHB�\)                                    Bxn�|  �          @�\)���R@c�
@���Bh{B�G����R��  @�{B��HC;�                                    Bxn�"  �          @�����@U�@��B}�
B��H��녿8Q�@���B�L�CT                                    Bxn��  �          A Q쿫�@G�@�Q�B��
B�{�����  @��HB��
CXǮ                                    Bxn�n  �          A���z�@S�
@�p�Bz��B�.��z�L��@�33B���CM��                                    Bxn  �          A Q��#33@HQ�@���Bn�C8R�#33�Q�@���B���CE�)                                    Bxn�  �          A���33@N{@ᙚBr�
B��33�Q�@��RB��)CG��                                    Bxn)`  �          A ���(�@\��@ۅBi�B��\�(���@���B��HC@O\                                    Bxn8  �          AG��G�@vff@�
=Baz�B����G���@��B�Q�C6��                                    BxnF�  �          Ap��?\)@��@�p�By�C���?\)��(�@��B�z�CQ޸                                    BxnUR  �          A���G�@(�@�G�Bv
=C���G��У�@�Q�B���CO��                                    Bxnc�  �          A��u�?�@��BlffCL��u�� ��@�(�Bk33CO��                                    Bxnr�  �          A
=�vff?�R@�
=By�C*��vff�R�\@�\)BT{C\�=                                    Bxn�D  �          A���HQ�?
=q@�p�B�Q�C*#��HQ��e@�\B`Q�Cd�                                    Bxn��  �          A���j=q>�33@�ffB��C.���j=q�i��@�G�BR�
C`�f                                    Bxn��  �          @��R�{��Y��@ۅBnffC@.�{���  @�ffB-(�Cc33                                    Bxn�6  �          @�Q���ff�333@�
=Bb�RC=� ��ff�z�H@�{B(z�C_{                                    Bxn��  �          A�����ͿB�\@љ�BY33C=E��������@�\)B!�RC\�
                                    Bxnʂ  �          @�p���녾�\)@�  Bg��C7������e�@�B4p�C]c�                                    Bxn�(  �          @�����?�@�(�Be\)C+�������2�\@�Q�BF�CV�{                                    Bxn��  �          A
�H��ff�L��@���Bl
=C4���ff�w
=@�=qB<33C\�                                    Bxn�t  �          A  ��z�@(�@�  Bd�\C(���z��{@��HBi{CL.                                    Bxn  �          A\)�z�?�G�A
=B��CǮ�z��5A=qB�� Cf                                    Bxn�  �          A����(�>�ff@ϮB?{C/�H��(��=p�@���B%p�CN                                    Bxn"f  �          A
�H���5@�G�B'Q�C:Q����fff@��HA�p�CQ8R                                    Bxn1  �          Az���\)���@��HB&��C=���\)�y��@��RA�Q�CS�                                    Bxn?�  �          A
=q���R���H@�{B7�C8�f���R�g
=@���B��CS8R                                    BxnNX  �          A33�˅��\)@�33B�C@!H�˅�|(�@w�A��HCS�q                                    Bxn\�  �          A(���Q�˅@��B$G�CB�=��Q����@s�
A�z�CV��                                    Bxnk�  �          A�
���
��
=@���B733C?�)���
���@�{B�CW��                                    BxnzJ  �          A������ff@��B,{C>)����s�
@�  A��CT                                    Bxn��  �          A���˅��ff@�G�B��C=h��˅�h��@~�RA�(�CQ�{                                    Bxn��  �          A
=���Ϳ�z�@g�A�{C@������S33@�RAvffCL�                                    Bxn�<  �          A���p��@c33Ař�CC��p��hQ�?�Q�AS�
CN�                                    Bxn��  �          Az���Q����@z�HA�Q�CC����Q��l��@A�p�CO��                                    BxnÈ  �          A33��33��@_\)AɅCD����33�hQ�?�\)AS�CO\                                    Bxn�.  �          Ap�����  @J�HA���CA�H���I��?�ffAMp�CK��                                    Bxn��  T          A���\)�У�@s�
A߮CA#���\)�W�@��A�CMǮ                                    Bxn�z  �          A\)�ڏ\��@���B��C8T{�ڏ\�333@fffA�CJE                                    Bxn�   %          A�\��=q�0��@��
B�C9� ��=q�7
=@W�A�(�CJ�                                   Bxn�  T          @�z���=u@l��A��\C3�������@K�A�\)CC&f                                    Bxnl  T          A����׿J=q@�33A���C:p�����<(�@S33A�p�CJ�                                     Bxn*  T          AQ��ڏ\���@��
B�
C8���ڏ\�8Q�@i��A�CJ�
                                    Bxn8�  T          A���녿�(�@�z�A�
=C?����Z=q@1G�A�p�CM��                                    BxnG^  
�          Ap���R�fff@hQ�AΏ\C:�)��R�*=q@(Q�A�
=CG�f                                    BxnV  �          A����(��8Q�@S�
A���C9\)��(��@��A�Q�CE
=                                    Bxnd�  
�          A�񙚿=p�@`��AƏ\C9�����p�@&ffA�\)CF\                                    BxnsP  �          A�H����G�@O\)A��HC9� ����
=@ffA�CD�3                                    Bxn��  T          A�R��33��@?\)A�=qC7Ǯ��33� ��@G�Az�RCBQ�                                    Bxn��  
Z          A���;��H@  A}�C7����Ϳ˅?�z�A:ffC?c�                                    Bxn�B  T          A\)���R��?���AN�\CB����R�%?
=q@u�CF�
                                    Bxn��  �          A���
=��33?��RA`(�CA����
=�*=q?333@��
CF�R                                    Bxn��  �          A�R��녿�Q�@33A�33CBh�����7
=?p��@��
CH��                                    Bxn�4  
�          A���=q�.{@�G�Bd�C<�q��=q��z�@�p�B(�C_�                                    Bxn��  T          AQ���p��+�@���B`��C<���p���Q�@�z�B&�C^aH                                    Bxn�  �          A
�H���
����@�Q�B9�C>L����
��ff@���B��CW�\                                    Bxn�&  "          A����ÿ
=q@�(�B0��C9������c33@�
=B��CRxR                                    Bxn �  T          A
�\�Ӆ���@��B��C90��Ӆ�[�@��A���COk�                                    Bxn r  T          A�����\)@��BC<���qG�@~{Aϙ�CO}q                                    Bxn #  �          A�R��
=�{@�\)A�(�CFE��
=��=q@33An�\CR\                                    Bxn 1�  T          A�������Q�@��A�=qC>xR�����e@C�
A���CL�R                                    Bxn @d  
%          Az���׿��@���A�33C?������c�
@5�A�=qCMW
                                    Bxn O
  
�          A����(���z�@�33Aޏ\C>z���(��W
=@/\)A�\)CK�                                     Bxn ]�  "          A���\�\)@~�RAمCF�q��\���R@z�A\  CQ޸                                    Bxn lV  
�          A	���녿�p�@c33A��HC@�H����U@
=Ad��CKٚ                                    Bxn z�  T          A  ��Q콸Q�@W
=A���C4����Q����@333A�\)CA��                                    Bxn ��  �          A���
��{@A�A��C6O\��
���@��Av=qC@��                                    Bxn �H  �          A�����ff@�RAx(�C<Q���=q?�
=AG�CC:�                                    Bxn ��  
�          A���Q��G�@�Af{C@�f�Q��>{?p��@�p�CF:�                                    Bxn ��  W          AG����2�\?�33A�RCF8R���G
=��z��CH#�                                    Bxn �:  T          A(����8Q�?�
=@���CG�����Dz���\(�CH�3                                    Bxn ��  �          A\)����  ?�AMp�CC�\����;�>�(�@8Q�CG�H                                    Bxn �  
�          A
=���$z�>�{@G�CE�
���
=��=q��  CD:�                                    Bxn �,  T          A  �{�O\)    <#�
CI���{�0  ���H�4��CF��                                    Bxn ��  
Z          A���=q�z=q>W
=?��CM���=q�[�����Ap�CJ�
                                    Bxn!x  
�          A��� z��Z�H�u��Q�CK�� z�����'���33CD��                                    Bxn!  
�          A� (��W���Q��/�CJ�{� (����H�N{��z�CAǮ                                    Bxn!*�  %          A�
��\)�U�z��v�HCKW
��\)�����o\)�ʸRC?��                                    Bxn!9j  �          A\)��=q�Z=q�)�����CLG���=q���R�������
C?!H                                    Bxn!H  "          A
�R��G��hQ��\)�o�CM�R��G�����u��ѮCB�                                    Bxn!V�  �          A����H�c33���=��CL^����H�33�Z�H���HCB��                                    Bxn!e\  �          A�\��Q��tz���\�T  CN8R��Q��	���p  ��  CC�                                    Bxn!t  �          A\)���
�x���'
=����CO���
��z�������ffCB�                                    Bxn!��  
�          A����Q��I���2�\��\)CJ)��Q쿘Q���G���C<�3                                    Bxn!�N  
Z          AG���\)�����P  ��33CA���\)>#�
�o\)�ȸRC2�
                                    Bxn!��  �          A���  ��(��E����CW�H��  �\)���R���CHE                                    Bxn!��  
�          A(���Q���ff�I�����HCR{��Q��������
CB�                                    Bxn!�@  
�          A����������_\)��(�CW
����(���  �G�CE�{                                    Bxn!��  "          A33��(�����}p�����CY���(����H��p��&33CE�                                    Bxn!ڌ  
�          A33�Ǯ��ff��=q��Q�CZff�Ǯ�G��\�+33CE�3                                    Bxn!�2  T          A����������(��ۙ�C[5�����33��p��.=qCFk�                                    Bxn!��  T          A33��ff��z��}p���=qC`�3��ff�*=q�˅�533CM�                                    Bxn"~  
W          A����\)���������=qCc�R��\)��R��ff�H�CMW
                                    Bxn"$  
+          AG����
���z���(�CCT{���
?���=q�  C/�R                                    Bxn"#�  �          A{�����ff��CE
��>��������\)C1                                    Bxn"2p  T          AG����H�E���\)���CKp����H���
��z���C4�H                                    Bxn"A  T          A����Q��o\)�����(�CQ�)��Q����  �.�HC833                                    Bxn"O�  
�          A��Ϯ�u���G���CR��Ϯ�
=q����0G�C8Ǯ                                    Bxn"^b  
�          AQ���33���H����(�CWu���33�xQ���33�;�C=
=                                    Bxn"m  	�          A�
��z��u��=q���CS  ��z�+���(��.�\C9�R                                    Bxn"{�  
Z          AG���\)�6ff�����=qCMz���\)?
=q��{�7z�C.�\                                    Bxn"�T  
�          Ap��˅�=q����
=CH���˅?u�����,�C+n                                    Bxn"��  "          A=q��33���R��\)�
=C@O\��33?����ffC&Q�                                    Bxn"��  
�          Az���=q�\��  �z�C6�q��=q@�R��z����
C!=q                                    Bxn"�F  
(          A33�љ���
=��
=�'�C>=q�љ�@(���p����C!z�                                    Bxn"��  
�          A	p���?#�
������C.���@<���\����z�C�)                                    Bxn"Ӓ  T          A	��G��\)��G���p�C5���G�@�
�z=q����C"h�                                    Bxn"�8  	�          A
�\��G��\)����G�C5.��G�@0  ���� ��C��                                    Bxn"��  
�          A�
�θR�.{����'\)C:�θR@"�\�������C�=                                    Bxn"��  �          AQ�������Q���\)�5(�C?!H����@���z��)(�C�                                    Bxn#*  
�          A����ÿ0���θR�?��C:�=����@8Q���=q�((�C�                                     Bxn#�  
Z          A(���ff��Q�����<\)C7u���ff@HQ�����G�C:�                                    Bxn#+v  �          A
�H��ff��������CffC5���ff@W����H� �Ch�                                    Bxn#:  
(          A
�\��33�
=���H�<�C9���33@9����p��#C�f                                    Bxn#H�  
�          AQ����ͿO\)��Q��H
=C<������@3�
���0��C��                                    Bxn#Wh  T          A���(��@  ���R�+=qCQ(���(�?����G�RC.c�                                    Bxn#f  "          A����z���
��p��+CB����z�?�  ��33�)Q�C#s3                                    Bxn#t�  �          A=q��ff?J=q��{���C,��ff@\���}p����C��                                    Bxn#�Z  
�          Aff���@����z�����C�����@�G����x��C�\                                    Bxn#�   �          A�H��z�>Ǯ��(��.��C033��z�@W
=��G��Q�CW
                                    Bxn#��  �          A  ��33?�Q���(��-(�C(xR��33@�����33��(�Cu�                                    Bxn#�L  
�          A	p���33?�(������ p�C"���33@���_\)��33C5�                                    Bxn#��  �          A\)��Q�@dz���=q���C���Q�@��
�#�
��\)C.                                    Bxn#̘  "          A
=��(�?�=q���R�Q�C&�\��(�@�  �U���\C�)                                    Bxn#�>  �          A\)��?�\)��(��G�C'�R��@q��XQ�����C��                                    Bxn#��  �          Ap���p�=�G�����2��C2�
��p�@G
=��p��33CE                                    Bxn#��  
�          A (����R��
=��p��!��C?+����R?����  �  C#L�                                    Bxn$0  �          @�z���p���  ��(��G�CD� ��p�?��������C)\                                    Bxn$�  �          @�����=q�'
=���R��CM&f��=q>����H�4  C/@                                     Bxn$$|  �          @��
�����=p���ff�Q�CP0�����<������4�\C3��                                    Bxn$3"  �          @�33����a���z����CWG������33�����I�C7�R                                    Bxn$A�  �          @�����p��O\)���R� Q�CU^���p��u��p��I�C4�                                     Bxn$Pn  �          @�  ��z��fff��z��1�C\����z����\)�e�\C5ٚ                                    Bxn$_  �          @����P  ���\���
�.�HCj�H�P  �xQ��߮=qCD��                                    Bxn$m�  �          @���S�
���\����-�Cj��S�
�z�H��\)�~�RCDh�                                    Bxn$|`  �          @�\)�{�J=q�ҏ\�jp�Cg���{?aG���  .C �                                     Bxn$�  �          @�ff��(���������D\)Cz(���(��^�R���(�CN��                                    Bxn$��  �          @�(����\�������\�4�RC�ff���\��ff���{Cf��                                    Bxn$�R  T          @�
=�=p���33��  �+�RC�AH�=p������{�Cx��                                    Bxn$��  
�          @�33����������+��C�XR����33���H�C�
                                    Bxn$Ş  �          @�\)?
=q��p��dz���p�C�5�?
=q�z=q��\)�np�C��                                    Bxn$�D  �          A ��    ��33�,(���{C�H    ���R���
�PffC��                                    Bxn$��  �          A=q    ���H�6ff���C��    ��=q��(��R33C��                                    Bxn$�  �          Aff?.{��
=���H��=qC��H?.{�����=q�v��C���                                    Bxn% 6  �          A�?s33��ff��z���C��?s33�<������C���                                    Bxn%�  �          A
�R�W
=��
=���cz�C����W
=���
�
=q±{CHh�                                    Bxn%�  �          A(�@��������H��C���@����
��=q��C�H                                    Bxn%,(  �          A
=@����=q����C���@�o\)����u�C���                                    Bxn%:�  �          @��R@vff�˅��z��I�C��R@vff���R���\�Q�C�e                                    Bxn%It  �          @��@]p������aG�����C��3@]p����R�n�R�	C���                                    Bxn%X  �          @�  �aG�=L�Ϳ�=q���C3B��aG�?c�
�����ffC%��                                    Bxn%f�  �          @�
=��{@ff����C�f��{@G
=�   ��G�CE                                    Bxn%uf  
�          @�=q��Q�?��ÿ�\)�f�HC#.��Q�@���\�XQ�C8R                                    Bxn%�  �          @׮���@L���q���HC�����@�����H�LQ�C�
                                    Bxn%��  �          @׮�w�@J=q��p��%�HC�=�w�@�G��G����HB�\                                    Bxn%�X  �          @�\)���
@   �I��� 33C�
���
@XQ��=q�vffCǮ                                    Bxn%��  �          @�
=����?˅�,(��י�C"}q����@2�\��z��[�C�                                    Bxn%��  �          @�����?E��Q����
C+�{���?�ff�����[�C!xR                                    Bxn%�J  �          @�����z��(���
�ŅCJ����zᾮ{�*=q���C8�f                                    Bxn%��  �          @��H�z���
==�Q�?G�C{�H�z���G��Q���{Cx��                                    Bxn%�  �          @�p���\��?�@�{Czh���\��ff�AG���\)Cx#�                                    Bxn%�<  �          @߮��  ��=q>�=q@  C�����  �����\����=qC�                                    Bxn&�  T          @�33�*�H��(�?�=qA6=qCp��*�H��\)�˅����Co=q                                    Bxn&�  T          @��
�L���^�R?Y��A#�Ccn�L���X�ÿ�z��_
=Cb��                                    Bxn&%.  �          @�p��QG��^�R?��AMG�Cb�\�QG��aG��u�4��Cc)                                    Bxn&3�  �          @!G���G���p�>k�@�(�C`����G��˅�=p�����C^\                                    Bxn&Bz  �          @��-p��@��?.{Ap�Cd��-p��:=q��ff�l(�Cb�q                                    Bxn&Q   �          @�Q������G�?k�@��Cep���������Q����RCc��                                    Bxn&_�  �          @��H�������?��HAb�HCgQ������zῴz��:{Cg�=                                    Bxn&nl  �          @�
=�vff���?��A�Cm@ �vff��(���
��\)Ck��                                    Bxn&}  �          @�p���
=��p�@z�Az�HC�h���
=��������HC�e                                    Bxn&��  �          @���n{��\)@%�A���C�q�n{��{��{�X  C��=                                    Bxn&�^  �          A �Ϳ�  ��{@$z�A��C�N��  ���Ϳ��U�C�u�                                    Bxn&�  �          @�33�|(���  �
=q��=qCfJ=�|(��5��(��({CW��                                    Bxn&��  �          @����=q�S33�8����Cp�3��=q��p����R�y\)CUǮ                                    Bxn&�P  �          @�(��������\)�G�Cq����ÿ�\����C@\                                    Bxn&��  T          A=q�
=��z���{�>\)Cu\)�
=��  ��\)u�CJ��                                    Bxn&�  T          A�R��\�g����
�uffCs�ÿ�\?k��ff�=Cz�                                    Bxn&�B  
�          A	�z��c�
��\)�q�\Cl���z�?��\���qC#�                                    Bxn' �  $          A
=q���R��\)��H�C_=q���R@5������3B�#�                                    Bxn'�  �          A	���\�����33��Cc�ῂ�\@fff��\)\)Bә�                                    Bxn'4  
�          A
�R��  �����(�.CfT{��  @dz���=qL�B�=q                                    Bxn',�  �          A
�H�W
=>���	p�¨G�C"�f�W
=@��H��33�]33Bǽq                                    Bxn';�  T          A(��^�R?�\)�	��B��)�^�R@�����\)�@
=B�.                                    Bxn'J&  
�          A
�H��ff?:�H���¨��B��)��ff@�
=�ڏ\�Q�HB��q                                    Bxn'X�  �          Azᾮ{@G��(���B�.��{@������H�0=qB�8R                                    Bxn'gr  �          A(�=L��?�p��Q��=B��)=L��@�z���=q�2�HB�                                    Bxn'v  �          A
=�c�
@C33�33��B�z�c�
@�\)�����=qB                                     Bxn'��  T          A�.{@Fff�(�(�B̞��.{@�{��Q��ffB��q                                    Bxn'�d  
Z          A���H�
=@�
AR�RC�>����H�
=�<(���p�C��                                    Bxn'�
  T          A�����=q@C�
A��C������\)��
=�BffC�R                                    Bxn'��  
�          AQ��\)��@�RAy�C��q��\)�
=�&ff���C��R                                    Bxn'�V  �          A���  �	p�@
=qA_
=C�Ϳ�  �ff�2�\����C���                                    Bxn'��  T          A�H�\)�(�?p��@�=qCB��\)���r�\�ʣ�C}�q                                    Bxn'ܢ  �          A����
�  >��H@L(�C�0���
�������RC~��                                    Bxn'�H             A
ff�ff�?\)@l��C��ff��33������{C~{                                    Bxn'��  $          AQ�����@33ARffC�����  �5�����C0�                                    Bxn(�  
�          A	� ����?�{AH��C��� ����33�2�\��{C��                                    Bxn(:  �          A��G��  >.{?�=qC�G���Q�����C|��                                    Bxn(%�  
�          A�<���\)�=q�w\)CzE�<����(���ff�5��Crc�                                    Bxn(4�  �          A���QG���{����Q�Cs�R�QG��Y�����
�a�
Cb&f                                    Bxn(C,  �          A����P����(��nz�Cj\��?p����CaH                                    Bxn(Q�  
�          A	G��i����(���G����Ck޸�i���ٙ����uG�CL�R                                    Bxn(`x  
�          A����R�h���ff�CSp����R@o\)��=q�x{B�p�                                    Bxn(o  
�          A\)��z����¢=qC9\��z�@�Q���
=�e  B�
=                                    Bxn(}�  T          Aff�QG��Q��
=.CB
=�QG�@l����33�`\)C��                                    Bxn(�j  "          AQ��n{�u���\�z�CBs3�n{@X�����
�[
=C	��                                    Bxn(�  
�          A33�HQ���{�fC4�)�HQ�@��\��(��P�B��3                                    Bxn(��  �          A
=�:�H��\)��HC5J=�:�H@��R�����U�B�ff                                    Bxn(�\  T          A
�H�B�\>��
�{�
C-��B�\@�(���ff�I��B��                                     Bxn(�  R          A33�fff?���\�C+���fff@�=q�Ӆ�>�RB�z�                                    Bxn(ը  V          A
�H�n{<��
����C3�3�n{@�{�ҏ\�F33C�\                                    Bxn(�N  �          A
�\�Dz�?��� Q�C p��Dz�@��R���6�B�                                      Bxn(��  "          A	��&ff���ff�\C4���&ff@��H�����W  B���                                    Bxn)�  �          A��1�?�Q���z�
=C�q�1�@�\)�����5�B��                                    