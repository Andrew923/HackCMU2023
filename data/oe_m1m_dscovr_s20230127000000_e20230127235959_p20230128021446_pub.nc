CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230127000000_e20230127235959_p20230128021446_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-01-28T02:14:46.936Z   date_calibration_data_updated         2022-11-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-01-27T00:00:00.000Z   time_coverage_end         2023-01-27T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        ~   records_fill         "   records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx_b   T          @陚�,��@����9������B�(��,��@�=q�k���{B��                                    Bx_p�  
�          @����>{@�����G��
�\B�#��>{@�  ��\)�S
=B��f                                    Bx_L  T          @�Q��^�R@�=q�����C �{�^�R@��������\B�
=                                    Bx_��  T          @�
=�G
=@��=p���z�B�\�G
=@�p�?��HA}B�p�                                    Bx_��  	�          @陚��
@���7����\B�(���
@�G����ͿG�Bٙ�                                    Bx_�>  T          @陚� ��@��R��{�5��B�� ��@�{�:=q��Q�B�#�                                    Bx_��  "          @�Q��J=q@��@  ��ffB����J=q@�ff��R���B���                                    Bx_Ȋ  
�          @�z����@�����;�C����@��?Y��@޸RCh�                                    Bx_�0  	�          @���}p�@�z���vffB��f�}p�@�=q?�(�A��RB��f                                    Bx_��  	.          @����HQ�@�  ��p��=p�B�ff�HQ�@\@z�A���B�k�                                    Bx_�|  �          @������@�p����
�#�Bۀ ���@Ӆ?ǮAG�
B���                                    Bx_	"  
E          @��-p�@��
>��H@��B�\�-p�@��@C33AЏ\B�z�                                    Bx_	�  
Z          @����@�������I��B�(���@�
=?��A%p�B��H                                    Bx_	 n  �          @�녿�{@Ϯ�9������B˨���{@�    <#�
Bɨ�                                    Bx_	/  �          @�Q쿪=q@˅�����=qBˮ��=q@�Q�?˅A^�HB�\                                    Bx_	=�  	�          @�G��aG�@�  @`��A�z�B�\�aG�@���@�  BS(�B�L�                                    Bx_	L`  �          @�=q��G�@߮?�z�A33B�Q��G�@��@z=qB�\BД{                                    Bx_	[  �          @�Q����@������+33B�ff���@�(�?��RA>=qB܏\                                    Bx_	i�  
Z          @�G���p�@�  �z����\Bнq��p�@�  ?Y��@�  BϸR                                    Bx_	xR  7          @���"�\@�{�{��=qB�  �"�\@���?
=@�{B�                                      Bx_	��  
E          @����}p�@��R�*=q��=qB����}p�@�=q�W
=��z�B�G�                                    Bx_	��  T          @陚�z�@�z��Q���{B֮�z�@�?B�\@�
=B�L�                                    Bx_	�D  
�          @���/\)@�녿�Q��7�
B�\)�/\)@ҏ\?���A+33B�=q                                    Bx_	��  q          @��H�QG�@\��H����B왚�QG�@�G�>�  ?��RB�.                                    Bx_	��  
�          @����4z�@Å�2�\���B��4z�@�
=�u�   B�=                                    Bx_	�6  �          @�G��(��@�  �#�
��\)B����(��@�  >L��?���B޽q                                    Bx_	��  
�          @����1�@�  �AG���{B���1�@ָR���
�!G�B���                                    Bx_	�  �          @陚��@У׿�
=�xz�B��f��@׮?aG�@�G�Bڳ3                                    Bx_	�(  "          @���\)@ٙ�?���A6�RB؀ �\)@�(�@��\BB�k�                                    Bx_

�  
�          @�R�.�R@�z�>\@@  B��.�R@�p�@A�AǮB�{                                    Bx_
t  
�          @����G
=@�\)��(���RB���G
=@�{?�z�A8  B�ff                                    Bx_
(  "          @��H��@�
=��p��?33B�{�H��@�G�?�A33B��                                    Bx_
6�  
�          @���U@���R���HB����U@ʏ\>�Q�@9��B�R                                    Bx_
Ef  �          @�{�K�@ƸR��p��^�HB�8R�K�@�(�?n{@�p�B���                                    Bx_
T  �          @��
�L��@��
�c33���
B����L��@Å��
=�33B�.                                    Bx_
b�  "          @�33�9��@��R�\(����B��9��@�(��p����z�B��                                    Bx_
qX  �          @��0  @�Q��%���
=B�.�0  @�G�=u>�Bᙚ                                    Bx_
�  �          @��3�
@�33������RB�=�3�
@Ϯ>Ǯ@L(�B��)                                    Bx_
��  �          @ᙚ�G
=@������
�jffB� �G
=@�  ?O\)@�(�B���                                    Bx_
�J  T          @ᙚ�I��@\��
=�<��B��I��@�z�?��AffB�L�                                    Bx_
��  �          @��J�H@�z��z��_33B�\�J�H@��?Y��@�33B�8R                                    Bx_
��  "          @߮����@�p��������Cu�����@�ff?xQ�A
=qCL�                                    Bx_
�<  "          @�����Q�@�{?�{AX��CxR��Q�@h��@W
=A�{C��                                    Bx_
��  T          @�����@��?xQ�Ap�C@ ����@�33@7�AÙ�C^�                                    Bx_
�  T          @�����@�{?�\)AX��C�=����@�33@dz�A�ffCB�                                    Bx_
�.  �          @�{�N{@�\)@R�\A�Q�B��R�N{@O\)@���B>�RC��                                    Bx_�  �          @�Q��l(�@�=q@%�A�p�B�p��l(�@u@�  B ��C�{                                    Bx_z  �          @�\)��
=@�=q?��\A*{Cff��
=@�z�@L��A�p�C	:�                                    Bx_!   �          @������@��\>��?��RB������@���@Q�A�  C@                                     Bx_/�  T          @�����@�\)����  B������@��R?���Ao33B�B�                                    Bx_>l  �          @߮��33@�p��u�   C�f��33@��\?���Aw�C��                                    Bx_M  �          @�{��Q�@�(���(��c�
B���Q�@��\?�{Ax��B���                                    Bx_[�  T          @�p��z=q@�>�z�@��B�{�z=q@��H@"�\A�ffB�
=                                    Bx_j^  �          @�
=�aG�@��>���@0  B����aG�@��@,��A��RB��                                    Bx_y  "          @޸R�h��@����  ��B�33�h��@���@Q�A��RB��q                                    Bx_��  �          @޸R�l��@��
>B�\?���B�u��l��@��@!G�A�  B�Ǯ                                    Bx_�P  �          @߮�QG�@��;L�Ϳ��B��QG�@�
=@G�A�  B�=                                    Bx_��  T          @�=q�^�R@��H��Q�=p�B�k��^�R@�(�@A�{B�p�                                    Bx_��  
�          @���XQ�@��
>�@qG�B��
�XQ�@�@7
=A��\B���                                    Bx_�B  T          @�Q���G�@�  ?�\)A��B��=��G�@�33@Mp�Aݙ�C@                                     Bx_��  �          @�\��(�@P��@
=qA�(�C����(�@��@QG�A���C                                     Bx_ߎ  �          @޸R��@\)@�A��Cc���@.{@p��B��CJ=                                    Bx_�4  
�          @�����\)@�\)@-p�A�33C
33��\)@3�
@�p�B��CQ�                                    Bx_��  �          @�(���
=@n�R@:�HA���C{��
=@G�@��B  Cn                                    Bx_�  �          @ۅ��(�@L(�@c33A���C޸��(�?\@���B&{C"��                                    Bx_&  
�          @�33��p�@`  @8Q�A�C����p�@�@���B�C�                                    Bx_(�  �          @�
=���R@$z�@4z�AȸRC�=���R?��R@g
=BG�C'+�                                    Bx_7r  �          @�����@��@R�\A��HC�f���?8Q�@x��B��C,�                                    Bx_F  �          @ҏ\���@7
=@L��A�=qCL����?�{@�=qB��C$5�                                    Bx_T�  �          @ҏ\��ff@7
=@[�A�\)C����ff?��
@���B%33C$�\                                    Bx_cd  T          @�z���33@7�@P��B 33C�)��33?���@�(�B(C"�                                    Bx_r
  �          @У��(��@���������B�p��(��@�(�?�\@��B�W
                                    Bx_��  T          @�=q���@y����{�k
=B��=���@��
��(��p�B���                                    Bx_�V  
�          @�����@�\)��p��8��B�{����@�33?fff@߮B��                                    Bx_��  �          @���u�@�\)�(����B�W
�u�@θR=���?G�B�aH                                    Bx_��  
�          @�ff�L��@���(��X\)B�W
�L��@�\)�x����B��                                    Bx_�H  �          @�(����@����{��\B�
=���@���   �u��B�W
                                    Bx_��  "          @�=q��@�����  ��B����@�33��\)�f{B�p�                                    Bx_ؔ  
�          @����@�{��{�B�녿��@�=q���B�ff                                    Bx_�:  
�          @�=q��\@�
=��
=��B���\@��Ϳ�ff�>{B���                                    Bx_��  
�          @�33����@�
=������RBٮ����@ᙚ�����n�\B�                                      Bx_�  
�          @�G���G�@���(��"(�B��ÿ�G�@�������p�BУ�                                    Bx_,  �          @�G��W�@�
=�u��
=B�G��W�@��ÿ�33�,��B�\                                    Bx_!�  "          @����33@�(��ff��(�B�Q���33@�33<�>�  B��                                    Bx_0x  "          @����AG�@Ϯ�333���\B���AG�@�=q�L�;�{B�G�                                    Bx_?  
(          @��G�@�
=�����ffB�u��G�@��>��H@n�RB�#�                                    Bx_M�  T          @�p����\@Ǯ��
=�P��B�k����\@��@�A|��B�B�                                    Bx_\j  �          @�z����@Å>��?���B������@���@'
=A�  C �\                                    Bx_k  �          @�33�\)@l(�@s33B��C	5��\)?�
=@�B@G�C0�                                    Bx_y�  �          @���|��@�@qG�B�HC^��|��@��@��HB?33CǮ                                    Bx_�\  �          @������@�ff@G�A�Q�C������@l(�@���B�\CW
                                    Bx_�  �          @�=q��z�@�z�@2�\A��C����z�@[�@��RBC��                                    Bx_��  �          @��H���H@�(�@XQ�Aڣ�C�H���H@L��@�  B+ffC��                                    Bx_�N  �          @陚����@W
=@���B%�\C�\����?�  @��RBP\)C#��                                    Bx_��  �          @��
��ff@���@G
=A�G�C&f��ff@0  @���B\)C�                                    Bx_њ  �          @�����{@��\@�A��
CaH��{@6ff@p��A��RCk�                                    Bx_�@  �          @�\��
=@��\?���Aw�C���
=@AG�@W�A��C5�                                    Bx_��  
�          @�G���@mp�@�\A��
C�R��@(��@S�
A��C�q                                    Bx_��  �          @������@a�@\)A��
C������@�
@j=qA�  C8R                                    Bx_2  �          @�33���@��\@#33A��\Cu����@AG�@���BQ�C{                                    Bx_�  �          @����(�@��@&ffA�33C�H��(�@P��@��Bp�C=q                                    Bx_)~  
�          @�  ��\)@��@n�RA�G�C� ��\)@G
=@��\B5��C�3                                    Bx_8$  �          @�{��\)@���@A�33C	k���\)@h��@qG�A��HC(�                                    Bx_F�  "          @陚��(�@���@A�A�Q�C
�R��(�@333@�{B�C&f                                    Bx_Up  �          @�\���@�p�@�A��HC���@P  @mp�A�G�C�q                                    Bx_d  T          @��H���\@��H@33A�  C�H���\@���@{�BC	�=                                    Bx_r�  "          @�=q����@�=q?��A)Cff����@mp�@>�RA�
=C{                                    Bx_�b  �          @��H����@�z�@
�HA���C	�R����@^{@qG�A�=qC
=                                    Bx_�  �          @�(���
=@��?�z�Ax  C\)��
=@j=q@c�
A��C�H                                    Bx_��  �          @���p�@�  ?�p�AXz�C#���p�@�ff@eA�ffC��                                    Bx_�T  
�          @�����33@�Q�@0��A�(�C�H��33@tz�@���B�\C
�                                     Bx_��  �          @�(���Q�@��R�����I��C�H��Q�@�
=?���AL��C5�                                    Bx_ʠ  �          @�Q���{@�
=���R�(�C\��{@�?��
A_33C��                                    Bx_�F  �          @��H���@��ÿp������B�B����@�{?�z�A,��B��                                    Bx_��  �          @�Q���Q�@�ff>�\)@
�HC�
��Q�@�{@z�A�Cc�                                    Bx_��            @�Q�����?�Q�@j�HB  C"B�����>8Q�@�G�Bz�C2                                    Bx_8  �          @�ff�����Q�@��BK
=C>� �����1G�@��B)G�CT��                                    Bx_�  
Z          @�ff�Mp��'
=@��BU�\C[&f�Mp���Q�@�B(�Cj�
                                    Bx_"�  
�          @��
�����/\)@�ffBy�Cj�{�������R@�{B+Q�Cx�=                                    Bx_1*  
�          @�׿޸R�0��@׮B�W
Cm޸�޸R���H@��RB0p�C{+�                                    Bx_?�  T          @�(��@���   @�(�Be�
C[�f�@�����@�  B#{Clٚ                                    Bx_Nv  �          @��
�7
=�R�\@���BV33Ce��7
=��Q�@��BG�Cq��                                    Bx_]  �          @�{�:�H�Tz�@�z�BO�RCd���:�H��{@\)Bp�Cp�f                                    Bx_k�  
Z          @��
�   �=p�@�Bc\)Ceٚ�   ���R@�z�Bz�CsE                                    Bx_zh  
Z          @�z�����*�H@��HB{�
Ck�q�������H@�(�B-ffCyc�                                    Bx_�  �          @�  �{�-p�@�
=Bn  Cc���{���\@���B%=qCr�                                    Bx_��  �          @�{�,���,(�@\Bg��C`���,����  @�z�B!Q�Cp^�                                    Bx_�Z  �          @�  �+��(�@ӅByz�C[0��+����@��\B633Co�                                    Bx_�   T          @���AG���@�33Bl��CY
�AG���
=@���B,p�Ck��                                    Bx_æ  �          @�\�j�H����@ÅBdz�CI���j�H�g
=@�z�B3�RC`�                                     Bx_�L  �          @�G���?:�H@�{B
=C,�����u@�z�BQ�C=��                                    Bx_��  �          @�
=���?�G�@�33B�\C'T{�����G�@�Q�B�C8p�                                    Bx_�  T          @�(�����?�\)@�33B33C$�3���ͼ#�
@���B�HC4
                                    Bx_�>  �          @�{���?��H@~{B(�C(Q��������@�z�B��C7E                                    Bx_�  �          @�
=��z�?�33@|��Bz�C(���z�\@�33B�C7��                                    Bx_�  T          @�=q��p�?�\)@�{B33C(�
��p��
=@�G�B33C9�3                                    Bx_*0  T          @�z���p�?��
@�  B ��C$5���p����
@�\)B*33C7Y�                                    Bx_8�  �          @�����@@��B(�C�����>u@�33B.ffC1p�                                    Bx_G|  �          @���=q?�@\(�A��
C/O\��=q�E�@X��A�Q�C:�q                                    Bx_V"  �          @�(���\)?���@p��A�(�C"����\)>�33@�B��C0��                                    Bx_d�  �          @��H��p�?�z�@�
=B��C!h���p�>k�@�(�B�
C1�H                                    Bx_sn  �          @�33��\)@�@w
=B�Cp���\)?8Q�@�p�B��C,�\                                    Bx_�  �          @�������@\)@q�A�G�CǮ����?p��@�p�B  C*O\                                    Bx_��  �          @�����
@*�H@h��A��
C�����
?�z�@��B��C'�                                    Bx_�`  �          @����@HQ�@>{A�\)CY����?�@y��B�HC!��                                    Bx_�  �          @�p����@���?�{A/�
CǮ���@n�R@:=qA�=qC&f                                    Bx_��  �          @������H@�z�?h��@��HC)���H@��@'�A��\C:�                                    Bx_�R  �          @ᙚ����@L(�@=p�Ȁ\C�R����?�33@z=qB{C :�                                    Bx_��  �          @�G�����@l��@1�A���C������@(�@{�B33C!H                                    Bx_�  �          @�33����@�(�?�\)AuC	O\����@h��@Z=qA�RC                                    Bx_�D  �          @�33���H@�  ?�(�A�\C)���H@�\)@;�A�{C
��                                    Bx_�  �          @�\)����@#�
@l(�B�C�{����?��@��B/�HC&p�                                    Bx_�  T          @�p���z��^{@�p�Bm\)C{}q��z���ff@���B(�C��)                                    Bx_#6  �          @�(���\)�hQ�@��Bf\)C|�\��\)��G�@���B��C�B�                                    Bx_1�  T          @�ff��p��L��@��B}(�C��;�p�����@��HB)��C���                                    Bx_@�  �          @�ff�����@���B���C]!H�����@�ffBHCr�)                                    Bx_O(  �          @��
�j�H�!G�@��Bi��C=Ǯ�j�H�0  @�{BG\)CX��                                    Bx_]�  �          @��p�׾.{@�  Bg�RC6���p����\@���BNQ�CSff                                    Bx_lt  
�          @��K��#�
@���Bz��C4��K��{@�\)B_CV��                                    Bx_{  "          @����?\)?��\@ȣ�B|��C!.�?\)���\@�
=By�CJ��                                    Bx_��  S          @�  �Z�H?��@�Q�Bl�C"\)�Z�H��{@�  BlG�CE�                                    Bx_�f  �          @����x��?c�
@�{B[C'{�x�ÿ�
=@�z�BX�
CD��                                    Bx_�  �          @�  ��z��u@��RB[G�C��
��z���Q�@p  B�
C�~�                                    Bx_��  �          @��
�����
@�p�BF{C�����Ǯ@\(�A�G�C���                                    Bx_�X  
�          @�ff�z��C33@�
=B}�C��׿z���G�@�  B,p�C��)                                    Bx_��  �          @�����;�@�Q�B��C��{������@�=qB4�HC�>�                                    Bx_�  �          @�R=��
�)��@�{B��RC�ٚ=��
���\@��HB?ffC�w
                                    Bx_�J  �          @�\)����=q@�{B�p�C.����z�@��BZ33C��=                                    Bx_��  �          @���\)�+�@�p�B�W
Cwh���\)��Q�@��\B8�C�e                                    Bx_�  "          @�33��\����@�  B�C<n��\�(��@�
=Bw=qChG�                                    Bx_<  
�          @����
>���@ٙ�B�B�C*� ���
�@�\)B��Ce�=                                    Bx_*�  T          @������R@أ�B��C])���`��@�p�Be�Cx                                    Bx_9�  �          @��
�O\)��\)@�33B�\)Cs�{�O\)�y��@��B_G�C�'�                                    Bx_H.  T          @�=q���Ϳ�
=@�=qB�� C]+������^{@��Bhz�Cx�R                                    Bx_V�  �          @�
=�QG����@�
=Bbp�CQ���QG��o\)@�
=B.{Cdٚ                                    Bx_ez  �          @�G��i�����@���Bd�CD5��i���A�@�p�B=�C[��                                    Bx_t   �          @ᙚ�}p���33@�  B]Q�C9�}p���
@���BD{CR8R                                    Bx_��  �          @��vff?�Q�@�B_�C"޸�vff�k�@�
=Ba��CAz�                                    Bx_�l  
�          @߮�R�\<��
@��
Bv�HC3���R�\��@�Q�B_��CTQ�                                    Bx_�  �          @�\)�w
=>��@���B`Q�C-޸�w
=�˅@�=qBT�CJaH                                    Bx_��  �          @޸R�qG���ff@��BcQ�C:�{�qG��=q@��BGffCT�)                                    Bx_�^  �          @�\)�u��L��@���Ba�C6��u��
=q@��
BJ��CQc�                                    Bx_�  T          @߮�K����H@�Q�Bq��CH�)�K��L(�@��BE(�Ca�                                    Bx_ڪ  
�          @�\)��G���@�{BN\)C:
��G���@��RB6ffCP�                                    Bx_�P  
�          @�\)�P�׿޸R@�33Bf��CP��P���g�@�p�B4��Cd                                    Bx_��  �          @ᙚ�"�\�+�@ϮB��HCB���"�\�5�@�(�Bd\)Cd
=                                    Bx_�  "          @�
=�<�Ϳ=p�@�{B=qCB!H�<���333@��\BW��C_k�                                    Bx_B  �          @�Q��2�\<#�
@���B��)C3�=�2�\�	��@���Bn�RCY��                                    Bx_#�  
�          @߮�$z�>�=q@�B�(�C-��$z��z�@���By�
CX�{                                    Bx_2�  �          @޸R�.�R@  @�\)Bi=qCxR�.�R>B�\@��B�{C/�R                                    Bx_A4  "          @�\)��\@���@�G�B8ffB��
��\@��@�
=B{p�C�{                                    Bx_O�  �          @�Q���\@�z�@��B1z�B� ��\@*�H@�=qBs��C��                                    Bx_^�  o          @�\)�%�@�33@��B6�B�Q��%�@Q�@�=qBsC	W
                                    Bx_m&  �          @�p����
@�@��B+\)Bۮ���
@A�@�\)BrffB��                                    Bx_{�  �          @�ff����@�  @���B!�B�W
����@Y��@�(�Bj=qB噚                                    Bx_�r  
(          @���\@�@�33B
�B֣׿�\@�  @�  BS
=B���                                    Bx_�  	�          @��ÿ�z�@ə�@L(�A���B�z��z�@�(�@�33B2��Bـ                                     Bx_��  �          @���33@�33@9��A��HB�aH��33@���@��HB'�HB�u�                                    Bx_�d  
(          @�Q��G�@Å@S�
A�p�B؀ �G�@�@�z�B5{B��                                    Bx_�
  �          @�{�H��@%@�\)BZQ�Cu��H��?\)@�Q�B|
=C)�                                    Bx_Ӱ  
�          @���
=@��@���B��RC��
=    @�Q�B���C4\                                    Bx_�V  T          @�����\@�\)@��
BOG�BՅ���\@	��@�ffB��B�W
                                    Bx_��  T          @�p���G�@�=q@�G�B*  B�녿�G�@N{@�{Bs33B�Ǯ                                    Bx_��  �          @�G����=���@��B�=qC0������@�=qB�p�Cg�=                                    Bx_H  �          @�(���33���@�33B�Cin��33�|(�@��
BUQ�Czn                                    Bx_�  �          @��H�'��~�R@�{BC33Cl���'���G�@vffB �Ct�3                                    Bx_+�  T          @�=q�.{�5�@��BdQ�Cb)�.{����@��B'{Co�f                                    Bx_::  T          @���R�W
=@�(�B_
=Ci�{��R��p�@��RB��CtaH                                    Bx_H�  �          @�ff����S33@ȣ�Be�HCk���������@��B#Q�CvO\                                    Bx_W�  �          @�R����=q@ڏ\B��\CiJ=������@�{BF�\Cx33                                    Bx_f,  �          @�
=�{��
@��B�C^��{��z�@���BKp�Cq�=                                    Bx_t�  �          @�ff�C�
�}p�@ָRB��
CE޸�C�
�E@���BX�
CaY�                                    Bx_�x  T          @��H�6ff��z�@��Bu�
CU�
�6ff�u@��BA�Ciff                                    Bx_�            @�(�� ���q�@�ffBH�RCln� �����\@|��B33Ct�                                    Bx_��  =          @��
��R�
=q@�p�Bw�C]
��R����@��B>{Cnc�                                    Bx_�j  T          @���  �z�H@�33B�G�CK���  �B�\@�{Bf(�Ci�\                                    Bx_�  
�          @��H��Q�@ə�B}Q�C^33�����@�G�BB��Co�                                    Bx_̶  �          @�������@�
=B�ǮC@�f���)��@�ffBqz�Cd��                                    Bx_�\  �          @��
�\)�!G�@�=qB�Q�CB
�\)�+�@���Bk{Cc{                                    Bx_�  =          @�33�33��
=@�
=B�� CX(��33�h��@�33BQ�
Cm�                                     Bx_��  
�          @��/\)��\)@�ffB���CJO\�/\)�Fff@�Q�BY33Cd��                                    Bx_N  �          @��1G���=q@��B��{C9���1G��
=@�\)Bn�
C\c�                                    Bx_�  
u          @�Q���
��
=@�  B��CcY����
�}p�@���BT�
Cu�\                                    Bx_$�  T          @��Ϳ�ff���@ٙ�B��3Cf�R��ff��
=@���BN  Cv�                                    Bx_3@  T          @��ff�7�@�z�Bs��Ci޸�ff��\)@�p�B3�Cv
                                    Bx_A�  9          @�{�7����R@��Bq(�CO�=�7��L(�@��BC�\Cd�                                    Bx_P�  �          @����?�p�@��
BX�C�f��녾�=q@�33Be�\C7�q                                    Bx__2  	�          @���c33?�p�@�\)BmG�Cn�c33�(�@��
Bu�RC=��                                    Bx_m�  
�          @�=q�B�\�Ǯ@ӅB�33C;ff�B�\��@��Bf�
CZ�
                                    Bx_|~  o          @�R�a�?�\)@��Bc��C�\�a녿   @�{BlG�C<�                                    Bx_�$  
�          @��333?!G�@�ffB��C'ff�333��ff@��B��{CQ                                      Bx_��  "          @�{�3�
?�@�G�B��RC(���3�
��(�@���B~�\CO�R                                    Bx_�p  "          @�p��U���=q@�\)Br��CF��U��<��@�33BN
=C]�                                    Bx_�  "          @�p��@  �
�H@\Bj��CW��@  �{�@��
B7�Ch�                                    Bx_ż  �          @�\)�XQ��z�@���Bn(�CN��XQ��b�\@��HBC�CbJ=                                    Bx_�b  T          @��q녿��R@�  Bf{CF+��q��E@�=qBC{C[E                                    Bx_�  �          @���u�z�H@ǮBfG�CB@ �u�5�@���BG�CXc�                                    Bx_�  �          @�(��HQ��(�@�(�Bhp�CY��HQ���Q�@��HB5  Ci��                                    Bx_ T            @����*=q���
@׮B��3C:�{�*=q�ff@ʏ\Br�C]z�                                    Bx_�  
�          @�p��
=@l(�@�\)B[p�B�.�
=?У�@��B��3CO\                                    Bx_�  �          @�\���H@~{@�BK(�B�{���H@	��@��
B���Cn                                    Bx_,F  T          @����c�
�B�\@ȣ�BU(�C\���c�
��G�@�=qB!ffCic�                                    Bx_:�  �          @����Q����R@��
B5�HCi���Q���{@���A�G�Cq!H                                    Bx_I�  �          @����g����
@�G�B�HCin�g����@VffA��HCo�
                                    Bx_X8  T          A ���O\)��p�@�B4  Ck33�O\)����@���A�z�Cr5�                                    Bx_f�  
�          A ���^�R���@�G�B9
=Cg!H�^�R���H@��A�p�Co@                                     Bx_u�  �          A ���#33�fff@�\)Bc  Cj���#33��{@�33B&Ct�f                                    Bx_�*  �          A ����`��@�G�Bh��ClE����@�{B+��Cvk�                                    Bx_��  "          A Q��O\)���@��HBFQ�Cgh��O\)��@��
B��CpQ�                                    Bx_�v  �          A ���~�R�i��@�B>��C^��~�R���@��HBp�Ch�                                    Bx_�  "          A{�\(����R@��B8�\ChW
�\(����@�G�A��\Cp�                                    Bx_��  
�          Ap��?\)�vff@�ffBS�Ch0��?\)���\@�G�B  Cq��                                    Bx_�h  "          A{��{��\)@�
=BT��Cwn��{��@��
B33C}=q                                    Bx_�  
�          Aff�E��U@�{B_33CcT{�E����@�p�B(=qCo�                                    Bx_�  
�          A�R�ff�>{@�{Bx�RCg���ff��{@�Q�B>��Ct��                                    Bx_�Z  
�          A(��AG��l(�@��HBX�Cf�f�AG���ff@�\)B (�Cp�R                                    Bx_   
�          AQ��k�����@���B2p�Cg��k���G�@�ffA��Cn��                                    Bx_�  T          A33���\��\)@�B0�Cb
=���\��ff@��RA�(�CjaH                                    Bx_%L  "          A  �������@��
B,�
Cch��������H@��A�=qCkE                                    Bx_3�  T          A  ��Q����@�=qB*Q�C]�q��Q����@�p�A��Cf��                                    Bx_B�  �          A\)��  ��p�@��RBffC^���  ���@^�RA�p�Ce��                                    Bx_Q>  
�          A\)��{��p�@�
=B'�
Cb�=��{����@~�RA�z�Cj.                                    Bx__�  
�          A
=���H�(��@��
BG�C;aH���H�Q�@�p�B5  CM\                                    Bx_n�  �          A ����=q?�z�@�B �C!�{��=q>�33@�Q�B-�C0��                                    Bx_}0  
(          A��=q@AG�@��B�RC����=q?���@��\B&  C$��                                    Bx_��  T          A{���@\��@�33Bp�Cz����@�@���B��C z�                                    Bx_�|  "          A�
���\@0  @�(�B#z�C����\?�{@�{B8�C(��                                    Bx_�"  �          A  ���@�H@���B2��C}q���?333@�
=BD��C,�=                                    Bx_��  "          AQ���?�{@���B;33C#xR�����@ǮBC�
C5�
                                    Bx_�n  �          A�����@(�@��HB3
=C����?333@�G�BEQ�C,�=                                    Bx_�  
(          AQ����@U@��B G�C{���?�(�@���B<  C":�                                    Bx_�  T          A�
���R@>�R@�(�B#\)CW
���R?�{@�  B;G�C%�R                                    Bx_�`  
Z          A
=��Q�@=q@�G�B4  C^���Q�?333@�\)BFG�C,aH                                    Bx_  "          A33��G�@G�@�(�B.{C:���G�?�
=@ȣ�BH�\C$)                                    Bx_�  
�          Aff��z�@y��@�\)BffCn��z�@33@�33BA��C�\                                    Bx_R  
�          A��l��@Mp�@�(�BQ�
C
�l��?���@�Q�Br�HC W
                                    Bx_,�  "          @��R�hQ�@��\@�G�B:�C�)�hQ�@�@�{Bd��CG�                                    Bx_;�  �          @��\�k�@�{@�{B133CE�k�@!�@�(�B\{Cn                                    Bx_JD  �          @����Y��@�  @��B-�B�=q�Y��@7�@��HB[�C�)                                    Bx_X�  �          @���O\)@���@�
=B({B�
=�O\)@L(�@��HBY=qCn                                    Bx_g�  
Z          @����n{@�  @��B��C 
=�n{@N�R@���BJ�
C
�q                                    Bx_v6  T          @�  ��{?&ff@��BWp�C+#���{�s33@���BU=qC@�                                    Bx_��  
�          @��
���;#�
@��BP��C6���Ϳ��
@�
=BC�
CH�3                                    Bx_��  
�          @�p���������@�z�BU\)C9
=�����33@���BE=qCL33                                    Bx_�(  �          @��
�|�Ϳ0��@�
=Bh�HC=�R�|���=q@���BR=qCSh�                                    Bx_��  T          @�z���p�>�p�@�33Bb=qC.���p�����@ƸRB[{CE��                                    Bx_�t  "          @�����  @A�@�p�BA�
C޸��  ?�33@ȣ�B_�RC ��                                    Bx_�  �          @����,��@�ff@��
B�B�Ǯ�,��@�(�@��B8\)B�u�                                    Bx_��  9          @�{�������@�z�B@Q�CB� ����7�@�=qB(�
CQE                                    Bx_�f  
�          @����z��33@��B'�CL\��z��c�
@���B	33CV��                                    Bx_�  
�          @��
����#�
@�(�B>Q�C5޸�����=q@���B3��CE�{                                    Bx_�  
�          @��H��  @X��@�\)B4�
Ch���  ?�=q@�BT�RC�                                    Bx_X  
�          @�����33@b�\@���B33Cz���33@{@�=qB*33Cs3                                    Bx_%�  
�          @�{��p�@7�@�=qBG�C���p�?�Q�@�z�B4C%(�                                    Bx_4�  T          @��\����?�=q@���B433C#c�����=�\)@��B=�\C30�                                    Bx_CJ  T          @���=q@�\@�ffB8�RC
��=q>�@���BG=qC.�{                                    Bx_Q�  �          A(���ff@��@�=qB3G�C����ff?��@�BA�C.B�                                    Bx_`�  
�          A(����
@  @�(�B+��C33���
?5@���B;{C,��                                    Bx_o<  �          A���p�?�z�@�z�B.��C#���p�>��@�(�B8  C2��                                    Bx_}�  T          A���z�?��@��\B6\)C �\��z�>��R@��
BB33C0�R                                    Bx_��  �          A���z�?��H@�=qB#�C#Ǯ��z�>�z�@��\B-��C133                                    Bx_�.  
�          A  ��?�ff@��\B	C$aH��?\)@�z�B�C/                                      Bx_��  "          A��ٙ�@E�@^�RA�{C���ٙ�@@��A�=qC"�3                                    Bx_�z  l          A���{?�z�@�\)A�G�C&����{?�@�Q�B�HC/��                                    Bx_�   
          A����H@�@u�Aۙ�C#�����H?}p�@�
=A�=qC,�                                    Bx_��  T          A����\)?�@g�A���C%�3��\)?^�R@~{A��C-+�                                    Bx_�l  "          A=q��  ?�{@Y��A��C'����  ?333@l(�A���C.�3                                    Bx_�  
�          A��?���@Z=qA��C)����>��@i��AυC0aH                                    Bx_�  
�          A���G�?�z�@�=qA��
C)���G�>�33@�G�A��
C1G�                                    Bx_^  T          A����{?�p�@k�A�=qC%\��{?�  @��A��C,W
                                    Bx_  
�          A	����ff@.�R@QG�A�33C�
��ff?�=q@vffA�33C&.                                    Bx_-�  �          A
�\��@���@eA�z�C0���@G�@�Q�A��C�R                                    Bx_<P  
Z          A
�\��=q@�  @<��A��C!H��=q@w�@���A��HC}q                                    Bx_J�  
�          A
�\�ָR@�ff@H��A�z�Cn�ָR@b�\@�z�A�G�C33                                    Bx_Y�  "          A	����
@�=q@EA�Q�Cff���
@j=q@��
A�RC�                                    Bx_hB  "          A����  @�\)@*�HA�  Ck���  @l(�@l(�A��CY�                                    Bx_v�  
Z          A
�\��ff@���@   A�{C�q��ff@qG�@a�A�Q�C��                                    Bx_��  �          A
�H�ָR@���@-p�A�  C���ָR@}p�@s33A��
Ch�                                    Bx_�4  T          A\)��\)@�ff@��A�=qC޸��\)@_\)@VffA�\)CB�                                    Bx_��            A33��=q@���@,(�A��C�R��=q@aG�@i��A�ffC�\                                    Bx_��  
          Ap�����@���@Dz�A��C޸����@Q�@\)A���CW
                                    Bx_�&  "          AG���  @�z�@6ffA��HCJ=��  @U�@p��A�{CW
                                    Bx_��  "          A�����@��R@�As
=C�����@b�\@N�RA�  C\                                    Bx_�r  
�          A��޸R@��@p�Ak�C���޸R@��\@R�\A�G�C�f                                    Bx_�  �          A�
��@���@"�\A��CB���@j=q@aG�A��
C�q                                    Bx_��  �          A�����@��\@8Q�A�=qCJ=���@aG�@u�A�=qCW
                                    Bx_	d  T          AG���  @��@c33A�=qC����  @h��@���A��C�=                                    Bx_
  �          A(���33@�  @��Aa�C���33@hQ�@E�A��HC�R                                    Bx_&�  �          A
ff��33@z=q?�p�A
=C�=��33@`��@�Ac
=C8R                                    Bx_5V  �          A����Q�@b�\=�G�?8Q�CxR��Q�@Z�H?p��@ə�C:�                                    Bx_C�  �          A(���=q@xQ�>aG�?\C޸��=q@n{?���@�Cٚ                                    Bx_R�  �          A���z�@fff?�R@�C��z�@W
=?�Q�A��CG�                                    Bx_aH  
�          A
=��\)@W
=?�\@^{C����\)@H��?��
A\)C�)                                    Bx_o�  
�          A=q���@:�H?W
=@���CxR���@)��?�  A$��C!B�                                    Bx_~�  
(          A
=���@I��=�\)>�G�C����@C33?J=q@�z�C�3                                    Bx_�:  
�          A�R��\@fff��\�`  C�)��\@g�>���@  Cz�                                    Bx_��  l          A33���\@G����
�#�
CE���\@B�\?0��@�\)C                                    Bx_��  T          A�H��G�@9��>�=q?�{C�{��G�@0��?n{@���C z�                                    Bx_�,  
�          Ap����@?\)�.{��{C�)���@Dz�<��
>\)C^�                                    Bx_��  
          A���@HQ쿁G���\)C����@Q녾�=q��{C��                                    Bx_�x  
(          A���33@�{?���AG�Cn��33@q�@G�A|(�C�q                                    Bx_�  T          A����@�=q?333@�Q�C���@s�
?���A/
=C��                                    Bx_��  
�          A����\@i��?�33@��C�=��\@R�\?�Q�AY�C�=                                    Bx_j  �          A33��Q�@q�>��H@Tz�CL���Q�@dz�?���A��C�\                                    Bx_  �          A
=����@�>#�
?�=qC�����@�G�?��@�RC�R                                    Bx_�  �          Az���@�33?:�H@�C�H��@u�?�\)A0��CxR                                    Bx_.\  �          A�
���@n�R?�  A33C�����@W
=@�\A_
=C�                                    Bx_=  T          A
=q��  @�Q�#�
�uCp���  @�z�?���@��C�                                    Bx_K�  
�          A	����@��
��z����Cp����@��?W
=@�z�C�                                    Bx_ZN  �          A���(�@s33?�\A@��C��(�@Tz�@#�
A��RC��                                    Bx_h�  �          A�H���
@�G�?�
=ATQ�Cs3���
@aG�@1G�A�C�R                                    Bx_w�  "          A
�R��z�@��?s33@�=qC.��z�@��H?��AJ{C                                    Bx_�@  
�          A
=q��R@�p�?8Q�@�=qC���R@���?�(�A8z�C�H                                    Bx_��  "          A
�H��\@�{>�{@�RC���\@���?��RA\)C{                                    Bx_��  
�          A
=q��=q@�z�>��
@	��CY���=q@~{?��HA ��CT{                                    Bx_�2  
          A
�H����@���>Ǯ@#�
C\)����@�33?�ffA
{Cn                                    Bx_��  �          A
{��p�@��?
=q@g�C����p�@���?�(�A��C��                                    Bx_�~  
�          A	������@�(�>\)?p��C=q����@�Q�?��\@ڏ\C��                                    Bx_�$  
�          A�����@�G�<�>L��C�q���@|(�?c�
@��CT{                                    Bx_��  �          A���Q�@��>.{?�C
=��Q�@�
=?��@�p�C�
                                    Bx_�p  �          A����
@��ͼ��
�\)CE���
@��?h��@�Q�C��                                    Bx_
  �          A����@�G�>W
=?���C���@���?�{@�p�C�                                    Bx_�  
�          A����@��
>���@�CxR���@}p�?�Q�A�HCp�                                    Bx_'b  
Z          A�
���@��
>aG�?��RC�q���@�\)?���@���C��                                    Bx_6  T          A
=��{@�z�>�z�?�p�C�
��{@��?���A
=C��                                    Bx_D�  
�          A�H�陚@�{<#�
=L��C#��陚@�33?aG�@��C�3                                    Bx_ST  �          A�H��Q�@���=L��>��
C����Q�@��?n{@��HC+�                                    Bx_a�  T          A����@��\>�\)?��C@ ���@�?�z�@��C!H                                    Bx_p�  �          A���(�@��=#�
>���C����(�@��?fff@��HC.                                    Bx_F  
�          A33��z�@��H��G��E�C\��z�@���?:�H@�Cp�                                    Bx_��  
�          A���\)@|(��8Q쿙��C:���\)@x��?!G�@���C��                                    Bx_��  :          A
=���@s�
>\)?xQ�C����@l��?k�@�\)CǮ                                    Bx_�8  �          A�H��\@hQ�W
=��Ck���\@fff?�@hQ�C��                                    Bx_��  �          A�H���@k����w�C����@n{>B�\?��\C�=                                    Bx_Ȅ  :          A�\��p�@z�H��R��\)C&f��p�@}p�>8Q�?�p�C�H                                    Bx_�*  
          A  ��33@p  >\)?uC�3��33@i��?c�
@�C��                                    Bx_��  �          AQ����@vff���W
=C5����@q�?=p�@�z�C�f                                    Bx_�v  �          A��陚@\)���Q�C\)�陚@{�?.{@�p�C�3                                    Bx_  �          A33�޸R@�=q������C+��޸R@���?5@�
=C�                                     Bx_�  
�          A�R�ٙ�@�Q�    ��Cs3�ٙ�@�p�?c�
@ǮC�R                                    Bx_ h  
�          A�R��=q@�\)>aG�?���C�3��=q@�33?���@�  CxR                                    Bx_/  
          A���ff@~{�B�\����C#���ff@{�?
=@�33CaH                                    Bx_=�  
Z          A����ff@g��#�
����C���ff@e�?��@s�
C\)                                    Bx_LZ            A����p�@l(��u��G�C�=��p�@h��?(��@�G�C�f                                    Bx_[   n          A�����@<(������C5����@@  ���
�#�
C�\                                    Bx_i�  
Z          A�����@E�8Q����C����@J�H��G��B�\C��                                    Bx_xL  :          Az����H@N�R�#�
���RC����H@S33�#�
�uC��                                    Bx_��  �          A����\@Tz�G����CQ���\@Z=q���\(�C�                                    Bx_��  
@          Ap����@c33�
=��=qC�����@fff=���?+�Cp�                                    Bx_�>  
�          AG���
=@>{�fff��ffC�3��
=@E���
��C33                                    Bx_��  �          A���=q@3�
�\�'
=C 5���=q@5�>#�
?�{C {                                    Bx_��  "          A���ff@0�׿(����(�C @ ��ff@5���c�
C�                                     Bx_�0  
          A�R��z�@2�\�@  ��G�C���z�@8Q�W
=��  CQ�                                    Bx_��  T          Aff��p�@P�׿W
=��CE��p�@W
=�aG��\C��                                    Bx_�|  �          A ����@?\)��33�   C\��@@  >aG�?��C��                                    Bx_�"  �          @�{��@'����\��ffC k���@0�׿��p��Cn                                    Bx_
�  
�          @�\)��(�@XQ쿌����ffC����(�@a녾��Q�C��                                    Bx_n  �          A{��
=?��Ϳ�33�Y�C(G���
=?������3
=C&                                      Bx_(  �          A����=q?��
=q�x  C&\)��=q@�Ϳ���L(�C#�                                    Bx_6�  �          A���(�@p���R���RC#Y���(�@(Q����h��C n                                    Bx_E`  �          A��?�{�)������C'ٚ��@z��z�����C$��                                    Bx_T  �          A�R��{@�\�ff��=qC"޸��{@+�����W\)C .                                    Bx_b�  T          A���?��
���z�C,^���?�Q����rffC)h�                                    Bx_qR  "          A���?(���Q����C/���?����{��C,
=                                    Bx_�  T          A �����R?u��
�m�C,����R?�������T��C*\)                                    Bx_��  
�          @�
=��Q�?��
�����Q�C,(���Q�?����
�H�}��C)�                                    Bx_�D            @����@��G���z�C$.��@p���\)�XQ�C!�=                                    Bx_��  n          @��
��@������f�RC".��@(�ÿ�G��1p�C��                                    Bx_��  
�          @�Q����R@�  �����$��C
Ǯ���R@�p�����aG�C	�)                                    Bx_�6            @�p��ۅ@!G��{��G�CǮ�ۅ@8Q�޸R�R�HC=q                                    Bx_��  
�          @�{��ff@�녿�{�Ap�C���ff@��ÿG����HCE                                    Bx_�  
Z          @���33@Tz��(��o
=CW
��33@g
=��{�$  CW
                                    Bx_�(  �          @���Ϯ@a��z�����CxR�Ϯ@xQ���G�C!H                                    Bx_ �  �          @�p�����@Z=q�=q���C������@qG����
�W�
Cz�                                    Bx_ t  "          @�����Q�@S�
�33��(�C���Q�@j=q�ٙ��L��C�H                                    Bx_ !  �          @�z��љ�@hQ쿵�+�C  �љ�@tz�E���=qC�q                                    Bx_ /�  �          @�����z�@mp����v=qC�
��z�@�Q쿯\)�$Q�C��                                    Bx_ >f  �          @�����@B�\���H�p��Ck�����@U���z��,z�CaH                                    Bx_ M  �          @�33��\)@?\)�#�
��{C5���\)@X���G��v�HCh�                                    Bx_ [�  "          @�p���(�@aG���R��=qC#���(�@vff�����B=qC�                                    Bx_ jX  �          @�ff�ȣ�@mp��{��Ck��ȣ�@�=q���Yp�C                                    Bx_ x�  
�          @����\@�p���p��pz�C���\@�{��  ��HC�)                                    Bx_ ��  T          @�z����H@�  ��\)��
B��{���H@�33�u��G�B��R                                    Bx_ �J  "          A33��p�@�녿����p�B����p�@���=u>��B�B�                                    Bx_ ��  "          @�����@�{��
=���B�aH����@�G������
B�u�                                    Bx_ ��  �          AG����H@�
=��(��b�RB�����H@ָR�^�R��ffB���                                    Bx_ �<  �          A�R���\@ʏ\�Q���p�B��
���\@�(���ff��RB�L�                                    Bx_ ��  "          A�R���
@��H���p�C �����
@�zῦff��B���                                    Bx_ ߈  
Z          A   ���
@�Q��)�����RC+����
@��
���@��C }q                                    Bx_ �.  "          A �����@�33�p��33B��3���@�(���
=��B�=q                                    Bx_ ��  "          Ap�����@�z����X(�B��)����@��
�Q���33B�
=                                    Bx_!z  m          A=q��(�@�33��33�5�B����(�@��ÿ\)�vffB�u�                                    Bx_!   
s          A�
���@��H��33�z�C ����@�\)�������B��                                    Bx_!(�  
�          A
=���H@��������C8R���H@�G���33�
=C�
                                    Bx_!7l  T          A�����R@�p���ff��G�Cu����R@�Q���k�C�q                                    Bx_!F  
�          A{��=q@��Ϳ�\)��ffC	���=q@�Q�W
=��C��                                    Bx_!T�  �          A��Å@�
=��ff���C��Å@�33�\�%�CB�                                    Bx_!c^  m          A�����@��ÿ�����C������@��;����p�CJ=                                    Bx_!r  �          A����@Ǯ�Q���C
=��@�G�>.{?�z�C�\                                    Bx_!��  T          A{��@Ǯ�z���=qC���@�  >��@7�C��                                    Bx_!�P  	          A����ff@љ���Q�(�B�Q���ff@�\)?p��@�p�B��H                                    Bx_!��  �          @�(��}p�@�33?�(�A-�B��
�}p�@�G�@�RA��B�L�                                    Bx_!��  �          A�����@Ӆ=��
?\)B�.����@�Q�?���@�{B��                                    Bx_!�B  �          A����ff@��=���?=p�CJ=��ff@��?��
@���C��                                    Bx_!��  �          A���z�@�  �Q����HCk���z�@��=�\)>��HC!H                                    Bx_!؎  "          Ap��Ǯ@�=q�=p���G�C���Ǯ@��
=#�
>�\)C��                                    Bx_!�4  �          A z����@��׿333��  CW
���@��\=#�
>��C�                                    Bx_!��  
�          A���  @�G��E���
=C����  @�33�#�
��=qCJ=                                    Bx_"�  �          A��
=@��\�(����p�C8R��
=@�(�=��
?��C�R                                    Bx_"&  "          A  ��@�33��  ��
=C���@�{�8Q쿢�\C
�H                                    Bx_"!�  T          A�
�˅@��H��33� (�CT{�˅@��R��p��%C��                                    Bx_"0r  �          A���˅@�Q쿡G��
�\Cs3�˅@�(����I��C                                    Bx_"?  �          A��  @�
=���H�=�C����  @�p��u���Cz�                                    Bx_"M�  T          A=q��  @�ff��
=�T��C����  @���
=� ��Ck�                                    Bx_"\d  
�          A=q��Q�@�p����H�W�
C���Q�@��Ϳ��H���C�H                                    Bx_"k
  T          AQ����H@�33���f�\C
���H@�����C�)                                    Bx_"y�  
�          A
�R�أ�@�\)�'���
=C\�أ�@�녿�33�K33C5�                                    Bx_"�V  	          A����@�Q��4z���
=C�)���@��
����b�RC��                                    Bx_"��  T          A�أ�@�z��;���ffC#��أ�@����(��d��C�                                    Bx_"��  
�          A
{��p�@�33�Fff����C\)��p�@�Q���R���C�                                    Bx_"�H  �          Ap����@�
=�K�����C)���@�(��"�\��C�3                                    Bx_"��  �          A
=��@����I����\)C:���@�� ����z�C��                                    Bx_"є  T          Az���G�@��
�P  ��{C����G�@�G��(Q�����C0�                                    Bx_"�:  �          A�
��
=@l(��h����(�C���
=@�p��E���  C)                                    Bx_"��  
�          A����@p���z�H��G�C����@����U���\CT{                                    Bx_"��  	          A��޸R@|(��qG����C� �޸R@��K���z�C��                                    Bx_#,  ;          A����@�
=�AG����HC33��@�33�
=�y��C
=                                    Bx_#�  "          Az���Q�@���J=q����C�
��Q�@�z��"�\��z�C�                                     Bx_#)x  �          A������@�Q��U����RC������@��/\)��(�C0�                                    Bx_#8  T          Aff���H@����dz�����Cff���H@���>{��{C��                                    Bx_#F�  	          A�\�陚@x���Tz�����C���陚@���0  ���
Cu�                                    Bx_#Uj  	          A�H���@s�
�`����G�CO\���@�Q��<����z�C��                                    Bx_#d  "          A33�陚@n�R�e�����C���陚@��B�\����C33                                    Bx_#r�  �          A(���R@_\)�h������C����R@}p��HQ����\C\                                    Bx_#�\  �          A����@HQ��}p����
CT{��@h���_\)��{C�                                    Bx_#�  �          A  ����@_\)�q���Q�C����@~{�QG���=qC��                                    Bx_#��  
Z          A\)��z�@`  �hQ���G�C�f��z�@}p��G���\)C�{                                    Bx_#�N  "          A\)��Q�@`���xQ��ͮC.��Q�@�  �XQ���G�C!H                                    Bx_#��  �          A�R��\)@u��^{���
C
��\)@�Q��;���C�                                     Bx_#ʚ  T          A�H���H@��H�>{���C�H���H@�ff����yp�CǮ                                    Bx_#�@  �          A\)��z�@u�q���  C����z�@���O\)���C�)                                    Bx_#��  �          A=q�ڏ\@Y����������C���ڏ\@~{�����C�\                                    Bx_#��  "          A=q��@S33��{��Q�C���@w
=�~{��z�C��                                    Bx_$2  
�          A�\��@J=q��=q��(�C���@l���w����C��                                    Bx_$�  "          A�\��z�@"�\�������
C �q��z�@E��z=q��ffCY�                                    Bx_$"~  �          Ap���\?�(����H����C%p���\@�R�s33���C!ٚ                                    Bx_$1$  
�          A���?��H�����
=C'n���@{�s�
��=qC#�{                                    Bx_$?�  
�          A�����?�ff������  C&����@�
�q��ʏ\C#:�                                    Bx_$Np  �          A\)��p�?�z���
=��(�C'^���p�@(��~{��G�C#��                                   Bx_$]              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_$k�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_$zb              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_$�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_$��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_$�T              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_$��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_$à              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_$�F              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_$��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_$�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_$�8              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_%�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_%�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_%**              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_%8�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_%Gv              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_%V              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_%d�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_%sh              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_%�  r          AG���\)��  ����,33C=xR��\)���
����/(�C7\                                    Bx_%��  
�          A�����\����\�;p�C?����\�����p��?33C88R                                    Bx_%�Z  T          Ap���  ���
��\)�Y(�CI�{��  �z�H��z��a
=C@=q                                    Bx_%�   �          AQ�������H�����\�\CF
=����(�������b�RC<W
                                    Bx_%��  "          A����
���
�߮�i\)CEB����
��ff���H�n�
C:L�                                    Bx_%�L  
�          A\)���R���\��Q��^{CC�{���R����ۅ�c(�C:�                                    Bx_%��  �          @��
���������$�C5n��>�ff�����#p�C/�f                                    Bx_%�  �          @����\)<���33���C3����\)?
=q��=q�p�C/B�                                    Bx_%�>  
�          @���˅����\)�ffC8L��˅=#�
��Q��G�C3��                                    Bx_&�  �          @�����H�!G������z�C9�����H��G�����  C4�R                                    Bx_&�  
Z          @�ff�\�(���=q��RC9��\������ �C4L�                                    Bx_&#0  T          @�{���Ϳ
=q��(��{C8ٚ���ͼ#�
����=qC4�                                    Bx_&1�  �          @�ff���H����������C=����H���H��=q��C8�H                                    Bx_&@|  	          @���ff�xQ���33��C<���ff��
=�����C7�H                                    Bx_&O"  m          A   ���R��
=��  �-=qC?�R���R�\)��33�1(�C9�
                                    Bx_&]�  "          A   ��p�����=q�;  CGO\��p���33����B{C@�                                    Bx_&ln  �          A���p�>W
=������C2=q��p�?.{��(����C.ff                                    Bx_&{  
�          A{�У׾�����Q�C8=q�У�=L�����\�(�C3�                                    Bx_&��  T          A  �љ�������33�Q�C>Y��љ��&ff���R��HC9��                                    Bx_&�`  "          A�R��(���  ��=q�	  C<�{��(����H������
C8:�                                    Bx_&�  
�          A=q��\)>�������Q�C2����\)?8Q������\C-p�                                    Bx_&��  �          Az���Q쿹�������33C?���Q�}p���G����
C<                                      Bx_&�R  �          A  ��Q�У�������=qCA\��Q쿗
=��{��{C=�                                    Bx_&��  
�          A�\����\�y�����CBW
�������=q��{C>��                                    Bx_&�  �          A ����(�>\)�Fff��(�C2�3��(�>���C�
��{C0Y�                                    Bx_&�D  �          A �����þ�(��0  ��  C7E���þ��1����
C5                                      Bx_&��  �          @����
=���1G���=qC8T{��
=����3�
��
=C6�                                    Bx_'�  T          @�\)��R�Q��,(����
C:L���R���0����Q�C8\                                    Bx_'6  
�          A Q���{�Y���z��p(�C:W
��{�!G��	���y��C8��                                    Bx_'*�  "          A Q����H��\)�G���C<n���H�aG�����(�C:�{                                    Bx_'9�  
�          @�\)��ff���
� ����
=C=���ff��G��(Q���z�C;��                                    Bx_'H(  T          A Q��ᙚ�����dz���G�C?\�ᙚ�}p��l(���\)C<�                                    Bx_'V�  
�          @�����
��p��xQ����C>&f���
�O\)�\)��C:�                                     Bx_'et  �          @�ff��=q��\)�u����HC?W
��=q�u�|������C;�q                                    Bx_'t  "          @�p���\)�\�w���ffC@�R��\)������Q���C=Q�                                    Bx_'��  �          @�(���=q����33���RC@:���=q�}p���
=��C<�\                                    Bx_'�f  	�          @��H��=q��z�������\)C@&f��=q�}p������   C<�=                                    Bx_'�  
�          @��H���Ϳٙ���{�=qCB�
���Ϳ�  ���H���C?�                                    Bx_'��  "          @������ÿ�(������Q�CCW
���ÿ��\��{�
�
C?p�                                    Bx_'�X  �          @�  ���H��
=��(��	CE�����H��(������CA��                                    Bx_'��  T          @�  ��=q��
=����
CE�H��=q��(����\��CA�{                                    Bx_'ڤ  �          @���z����ff��CDh���z`\)���
�  C@�\                                    Bx_'�J  �          @��У׿�p��r�\��=qC@�\�У׿���z�H��G�C=xR                                    Bx_'��  
(          @�z���{��ff�a���  C<���{�0���g���  C9�)                                    Bx_(�  
�          @����љ�����r�\�홚C=z��љ��333�xQ����C:!H                                    Bx_(<  �          @��
���׿��
��(���RCB@ ���׿�=q��Q����C>(�                                    Bx_(#�  
�          @�\��{��\�����CEB���{������\���C@�q                                    Bx_(2�  
t          @�  ����=p�����#��C;�����u��ff�%��C6}q                                    Bx_(A.  
@          @����\)������=q�0�RCE
=��\)�����ff�6ffC?z�                                    Bx_(O�  �          @����׿�p�����933CF�����׿�z���ff�?�\CA                                    Bx_(^z  �          @�����
=�33�����4CL����
=�޸R��
=�=\)CG@                                     Bx_(m   �          @����{�&ff��=q�.G�CO���{��
�����8  CJ��                                    Bx_({�  
�          @�����=q�Vff��33�$  CV�
��=q�5�����0G�CRs3                                    Bx_(�l  n          @�����ff�S�
����CU����ff�3�
��G��+�CQ�=                                    Bx_(�  �          @�������^{��Q���CW������=p����\�,ffCS^�                                    Bx_(��  
�          @�G���\)�l����{�'�C[����\)�J�H�����4��CWG�                                    Bx_(�^  �          @��H�z=q�HQ���{�BCZ�R�z=q�#�
��
=�O��CU:�                                    Bx_(�  "          @��qG��j�H����8�
C`B��qG��G���=q�G�
C[��                                    Bx_(Ӫ  
t          @�\�aG�������(��*�HCf���aG��q������;Cc�                                    Bx_(�P  �          @��]p���  ��=q�&��Chn�]p��\)��\)�833Ce
=                                    Bx_(��  T          @�(��AG���  �����)��Cm�=�AG���\)���H�<=qCjs3                                    Bx_(��  "          @��Fff�����=q�'ffClٚ�Fff��\)��Q��9��Ci�=                                    Bx_)B  n          @��Fff������
=�$z�Cl�3�Fff���������6�Ci��                                    Bx_)�  
r          @�  �N�R����(���Cl���N�R���R���\�*�
Cj
=                                    Bx_)+�  T          @�{�P  ��(���=q�!Cj��P  ��z�����3ffCg��                                    Bx_):4  �          @����Fff���������#�Ck���Fff��=q���5p�Ch��                                    Bx_)H�  �          @��
�Y���<(����\�/�
C\�
�Y��� �����H�=�CXxR                                    Bx_)W�  "          @�G���\)@p  �����C�H��\)@}p���
��  C�                                    Bx_)f&  �          @��
����@q���R��G�C������@\)�
=��C�                                    Bx_)t�  
�          @�z���ff@~�R������C����ff@�ff��
����C
:�                                    Bx_)�r  
�          @љ���(�@�������(�C
�{��(�@������~�\C	�=                                    Bx_)�  �          @�=q����@J=q�Fff����C�)����@\(��2�\��Q�C\)                                    Bx_)��  �          @У����R?:�H����A\)C*.���R?�
=��
=�<��C$ff                                    Bx_)�d  �          @����\)?��������2\)C&=q��\)?\���,\)C!G�                                    Bx_)�
  T          @�
=��p�?�(�����*p�C�3��p�@����!�HCW
                                    Bx_)̰  
�          @У���G�?����
=�#G�C����G�@
�H�����Cs3                                    Bx_)�V  �          @���Q�@<(������
=C����Q�@S33�o\)���Cz�                                    Bx_)��  T          @�ff����@K��Z�H�Q�Cu�����@_\)�G
=��z�C�                                    Bx_)��  T          @У���{@HQ��N{���CB���{@Z�H�:�H�ՙ�C�                                    Bx_*H  �          @Ϯ��
=@@���33��
=C���
=@Mp�� ����p�Cu�                                    Bx_*�  "          @Ϯ����@\)�����z�C�H����@����=q�a��C�\                                    Bx_*$�  �          @�
=��z�?�z��{��C"  ��z�@���
=�pQ�C �{                                    Bx_*3:  �          @�G���G�?�\)����ffC"���G�@��(����HC =q                                    Bx_*A�  �          @����ff@������=qC����ff@'
=��z���33Cff                                    Bx_*P�  n          @����H@.{��G��Y�C����H@6ff��  �4z�C�q                                    Bx_*_,            @�����p�@7
=����p�C=q��p�@@�׿���_
=C�q                                    Bx_*m�  	�          @�����@���H��=qC��@"�\������C��                                    Bx_*|x  "          @�{��p�@��p���\)C W
��p�@{�G����RC�H                                    Bx_*�  �          @�{���R@G
=�ff����C0����R@R�\�����HCǮ                                    Bx_*��  �          @���33@0�׿��R�W33C�q��33@8�ÿ��R�1C��                                    Bx_*�j  
�          @�z����\@5��
=�+33C  ���\@<(��k���C:�                                    Bx_*�  �          @��
���R@H�ÿu�
{C����R@N{�+�����C�                                     Bx_*Ŷ  �          @�z���p�@S33�#�
����C����p�@Vff��{�Dz�CL�                                    Bx_*�\  �          @��
���@O\)��{�EG�C�H���@U�����CǮ                                    Bx_*�  "          @�33���\@W
=�333���
C����\@Z=q�����g�Cc�                                    Bx_*�  
�          @�33��{@J=q�c�
� z�Cٚ��{@N�R�����CT{                                    Bx_+ N  �          @����=q@4z�:�H�ָRC)��=q@8Q�����
C��                                    Bx_+�  "          @ə���{@H�þ�����C�q��{@H��>�?�C��                                    Bx_+�  �          @�����  @U?333@�ffC����  @P��?�  A�\C#�                                    Bx_+,@  T          @�33���
@S33?#�
@���Ck����
@N�R?n{A33C��                                    Bx_+:�  T          @����
=@S33>�33@J�HC�)��
=@P��?&ff@���C5�                                    Bx_+I�  �          @�
=���@X��=�?���CE���@W�>�
=@p��Cs3                                    Bx_+X2  n          @�Q���=q@U���ͿfffC���=q@U>B�\?��HC\                                    Bx_+f�  :          @У���(�@g
=?=p�@�G�C#���(�@a�?��AG�C��                                    Bx_+u~  �          @У���{@b�\>8Q�?�\)C����{@`��>��H@��
C33                                    Bx_+�$  l          @�ff��p�@[�=L��>�
=C�f��p�@Z�H>�33@G
=CǮ                                    Bx_+��  T          @Å�U�?�{@.�RB�RC�
�U�?�{@5B�RC!�=                                    Bx_+�p  n          @�ff��G�?}p�@�(�B�B�B����G�>��H@�ffB���C��                                    Bx_+�  :          @�  ��G���@��HB��{CD�=��G���R@��B�\)Cj�f                                    Bx_+��  	�          @�G��s33?n{@��B�.Ck��s33>��H@��B�(�C��                                    Bx_+�b  
t          @�������@��?���A<��C�����@|(�?�ffAuG�C��                                    Bx_+�  :          @ƸR�s�
@������!G�C(��s�
@�Q�(����{C ��                                    Bx_+�  
�          @ƸR�W
=@��?xQ�A(�B��R�W
=@�Q�?�z�AR�RB���                                    Bx_+�T  <          @�=q�`��@��?�z�AZ{B�{�`��@���?�A��RB�Ǯ                                    Bx_,�  �          @�Q���p�@��?�ffA#\)CǮ��p�@|��?�33AZ�RC�=                                    Bx_,�  
�          @������@���=�?�33CE���@��>��H@�
=Cs3                                    Bx_,%F  T          @��
��@�=q�n{�G�C	u���@�(��z���ffC	                                      Bx_,3�  �          @�(���=q@j�H��  ��G�C@ ��=q@s�
��
=�X(�C5�                                    Bx_,B�  �          @Å�l(�@�?�z�A0��C E�l(�@��?ǮAo
=C �q                                    Bx_,Q8  
Z          @\�fff@�
=?�  A=B����fff@�33?�33A|��C �                                    Bx_,_�  
�          @\�I��@��R?��Ak�
B��f�I��@���?�(�A�33B��=                                    Bx_,n�  �          @Å�r�\@��\?��A!G�C�
�r�\@�\)?���A^{CB�                                    Bx_,}*  �          @Å���\@�Q���uC
����\@�  >��
@A�C
�)                                    Bx_,��  �          @�=q��(�@�z�>aG�@z�CB���(�@��?��@��C}q                                    Bx_,�v  T          @�(��S33@�����
�.{B�(��S33@�z�>�(�@\)B�W
                                    Bx_,�  
�          @���|��@��R�#�
��{C  �|��@�ff>�p�@\(�C{                                    Bx_,��  l          @�=q��Q�@dz�?\)@�z�C(���Q�@`  ?^�RAG�C��                                    Bx_,�h  T          @�Q���G�@P  =�G�?��\C:���G�@N�R>Ǯ@k�Cff                                    Bx_,�  "          @��R���\@\�;��R�?\)CxR���\@]p��#�
��C^�                                    Bx_,�  T          @���=q@n{�����x��C����=q@o\)���
�G�C�q                                    Bx_,�Z  T          @�\)��G�@HQ��G���ffC.��G�@J=q�.{����C�R                                    Bx_-   T          @�\)���@Z=q�B�\���C����@Z=q=���?s33C��                                    Bx_-�  �          @������\@l(��W
=�=qC33���\@o\)����G�C�                                     Bx_-L  
�          @�(��tz�@��\?�=qA#�CǮ�tz�@�
=?�(�A_�
Cu�                                    Bx_-,�  n          @��U�@�  @(�A���B�\�U�@���@%AŅB�u�                                    Bx_-;�  :          @�(����\@�{@fffB33B��쿂�\@��
@}p�B,Q�B�Ǯ                                    Bx_-J>  
�          @�녿5@�@   Ạ�BÅ�5@�ff@<(�A�B�=q                                    Bx_-X�  �          @��
�Ǯ@�ff@0��A�ffB�.�Ǯ@�{@N{A���B��{                                    Bx_-g�  
�          @ə���@�
=@$z�A��HB�����@�\)@C33A�  B��                                    Bx_-v0  �          @ə�=u@�p�?�A|(�B�Ǯ=u@�  @
�HA�\)B��q                                    Bx_-��  
�          @ƸR���\@�Q��\���B�8R���\@�G�<#�
=���B�(�                                    Bx_-�|  �          @��׿�{@�G�>W
=@�B�Ǯ��{@�  ?5@���B��                                    Bx_-�"  "          @�����@�{=�?��HB�\)���@��?(�@�33B�ff                                    Bx_-��  T          @�=q=���@��    �#�
B���=���@�G�>��H@��B��                                    Bx_-�n  
�          @�����@��\�Q��G�B�aH��@�zᾸQ��a�B�                                      Bx_-�  
�          @�33�Z�H@�����=q�K�B��H�Z�H@��Ϳh���\)B���                                    Bx_-ܺ  
�          @��
�i��@��@\)A��C��i��@��@'
=AɮCQ�                                    Bx_-�`  �          @���g
=@���?�G�AhQ�C c��g
=@��?��A���CL�                                    Bx_-�  
�          @�(��*�H@�(�?�{AyG�B�\�*�H@�\)@33A�\)B�                                    Bx_.�  
�          @��H�1G�@�{?�Q�A���B���1G�@���@
=A��B�L�                                    Bx_.R  :          @�=q�[�@��
?�
=A�=qB�#��[�@�{@z�A���B�\)                                    Bx_.%�  
�          @��H�J=q@�\)@
=A��B���J=q@���@ ��AÙ�B���                                    Bx_.4�  
          @���Q�@�  ?�G�A�  B�=q�Q�@��\@
=qA�(�B�.                                    Bx_.CD  �          @�=q���@��
?�33A1�B�����@�  ?�{Axz�B��                                    Bx_.Q�  �          @����(�@���?�ffAo�B噚�(�@��
@   A�p�B��f                                    Bx_.`�  T          @�G���
@�p�?�=qAs�B݊=��
@���@�\A��\Bި�                                    Bx_.o6  T          @����U�@�G�?��HA`��B����U�@���?�\)A��\B�L�                                    Bx_.}�  
�          @�p��`��@�?���A7
=C\�`��@�=q?��HAs
=C�\                                    Bx_.��  "          @�  �s�
@Z�H�}p��/33C

�s�
@_\)�0����z�C	�                                     Bx_.�(  
�          @�33�>�R@o\)�ff��G�C �\�>�R@{��G���(�B�ff                                    Bx_.��  �          @����B�\@]p��C33�{CT{�B�\@mp��/\)��{C\)                                    Bx_.�t  
�          @�
=�W�@P  �3�
���C��W�@^�R�!G��ۮC&f                                    Bx_.�  �          @����[�@H���AG��
=C	��[�@X���/\)����CW
                                    Bx_.��  T          @�\)�U@^�R�%��=qC�{�U@l(�����\)C&f                                    Bx_.�f  
�          @����<��@hQ��8Q����RC��<��@w��#33�ݮB��R                                    Bx_.�  
�          @���s�
@XQ���
��=qC
n�s�
@aG���p��33C	E                                    Bx_/�  
�          @��H��33@2�\�������C�\��33@<(���������Ch�                                    Bx_/X  
�          @�(��e@R�\����G�C	�=�e@]p������{C
=                                    Bx_/�  T          @����@����
=���
B�
=��@����(����B���                                    Bx_/-�  �          @����Q�@��R�  ��  B�33�Q�@�z��{����B�\)                                    Bx_/<J  T          @����p�@��������B����p�@����   ���B��f                                    Bx_/J�  S          @�Q��
=@�G�� ����
=B홚�
=@���Q���  B�k�                                    Bx_/Y�  
�          @�G��h��@W
=������
C	O\�h��@X�þ���?\)C	�                                    Bx_/h<  "          @�
=��Q�@?\)�#�
���
C����Q�@>�R>�  @1�C\                                    Bx_/v�  =          @��R�l(�@e��0����z�C޸�l(�@g��\����C�                                     Bx_/��  9          @��R�e�@j=q=�\)?L��CT{�e�@h��>Ǯ@��
C}q                                    Bx_/�.  �          @�Q��HQ�@c�
�����{�CQ��HQ�@dz�#�
�.{C33                                    Bx_/��  =          @���$z�@��R��\��B�q�$z�@��
��z���\)B��
                                    Bx_/�z  
�          @�  �n�R@g
=��=q�;�C��n�R@g�=L��?\)C�)                                    Bx_/�   "          @�33���@AG�>��@�G�C@ ���@>{?:�H@�z�C�R                                    Bx_/��  
�          @��j�H@x�ÿE��=qCY��j�H@|(���
=��
=C�R                                    Bx_/�l  
�          @��s�
@k������;�C��s�
@p�׿E��\)CaH                                    Bx_/�  �          @������@P�׿���C33C����@U�Y����C33                                    Bx_/��  
�          @������@^�R�}p��(��C�����@c33�.{��
=C
�                                    Bx_0	^  
�          @�=q�k�@k�����8��C���k�@p�׿:�H��ffCY�                                    Bx_0  
�          @����=q@��\��
����B�#��=q@������
=B�W
                                    Bx_0&�  
u          @��$z�@�{�޸R��(�B�#��$z�@��\����eB��                                    Bx_05P  9          @����5�@�
=��p��zffB����5�@��H��=q�5p�B�\)                                    Bx_0C�  
�          @�G��G�@�G��\��
B�.�G�@�������=B��3                                    Bx_0R�  
Z          @����:�H@��R��p��O�
B�p��:�H@����Tz���B�\)                                    Bx_0aB  
Z          @����c�
@��
?�A��C�\�c�
@|(�@��A�(�C{                                    Bx_0o�  �          @��\�^�R@��@ffA���C�f�^�R@}p�@p�A�  CQ�                                    Bx_0~�  �          @��
�Vff@��?��As33B��=�Vff@���?�Q�A�ffB�u�                                    Bx_0�4  "          @�=q�Tz�@�z�?��A,��B�#��Tz�@���?�  Ao
=B��                                    Bx_0��  "          @��R�S�
@�=q?^�RA(�B��
�S�
@�\)?��
ANffB�                                      Bx_0��  
�          @�ff�N{@�33?^�RA��B����N{@�  ?��
AP��B�#�                                    Bx_0�&  T          @���-p�@�
=��G����
B�q�-p�@�  �#�
�\)B�                                    Bx_0��  
�          @�{��
@�p���ff���RB���
@�G���\)�?33B�                                    Bx_0�r  �          @�=q���H@�����\���
B�33���H@�=q��{���B�=q                                    Bx_0�  �          @�����p�@��׿\(���HB�W
��p�@��\��ff��p�B���                                    Bx_0�  �          @������@��?�@��B�z����@�p�?�  A"=qB�                                    Bx_1d  �          @���(�@���?�\)A0��B��H�(�@�?���A}��B��
                                    Bx_1
  �          @����!G�@�=q?xQ�A(�B����!G�@��R?�
=Af�RB���                                    Bx_1�  "          @�p��\��@�{?�\)A-��B��\��@�=q?��Ap��B�33                                    Bx_1.V            @�ff�Z=q@�?��A,z�B�G��Z=q@��?��HAn�\C aH                                    Bx_1<�  T          @����+�@z=q��ff��
=B��
�+�@{���Q쿐��B��                                     Bx_1K�            @��H�>�R@;��z���
=Ck��>�R@G
=��ff���C�q                                    Bx_1ZH  T          @�G����@8Q��W��;
=B��쿧�@K��E�'�B�q                                    Bx_1h�  
�          @����(�@�Q�L�Ϳ
=qB�(�@�  >�33@�G�B�Ǯ                                    Bx_1w�  "          @�=q����@�
=������z�B�{����@��\��ff�H  B�k�                                    Bx_1�:  
C          @��׿޸R@z�H��{��  B��f�޸R@�=q���R���B�B�                                    Bx_1��  �          @�33�0��@�(�?E�A&�RB��
�0��@�G�?�33Az�RB�B�                                    Bx_1��  "          @�p��5@��@A��B�녿5@tz�@-p�B�B��                                    Bx_1�,  �          @�33��
=@���?��
A�B�\��
=@��H@
�HẠ�B�z�                                    Bx_1��  �          @�=q��z�@�33?�(�A��HB�p���z�@�p�@�A�  B�=q                                    Bx_1�x  T          @�녿�z�@�  ?��Ao�B�{��z�@��?�(�A�33B�k�                                    Bx_1�  �          @�33��@�p�?�  A4Q�B뙚��@��?�z�A�(�B��
                                    Bx_1��  �          @��׿���@�
=?��A��B��H����@�G�@33A��HB�8R                                    Bx_1�j  �          @�녿+�@�{@z�AŮB���+�@�\)@�RA�ffB���                                    Bx_2
  �          @���z�@��@  A�
=B¸R�z�@�Q�@*=qA�  BÅ                                    Bx_2�  �          @�  �
=q@�{?aG�A)��B��
=q@��H?��\Av�\B�                                    Bx_2'\  �          @�p��0��@|(�?Q�A�HB�
=�0��@u?�Q�Aa�B�\)                                    Bx_26  �          @��I��@n�R>L��@�\C:��I��@l(�?��@�ffC�                                     Bx_2D�  T          @�=q�I��@a녿
=q��
=C�3�I��@dz�L���Q�Ck�                                    Bx_2SN  
�          @��G�@s33�:�H�p�B���G�@vff��Q����B�p�                                    Bx_2a�  o          @��ÿ�(�@P��?:�HA-p�B�B���(�@K�?��Axz�B�                                    Bx_2p�  �          @��H�G�@=q>\)@�\C@ �G�@��>�p�@���C�=                                    Bx_2@  
�          @�z��U?���A��33C��U@
�H�5���RC��                                    Bx_2��  "          @�{�O\)@\)�'
=�33C}q�O\)@.�R�ff���C��                                    Bx_2��  �          @�{�   ?���~{�Y�HC�
�   ?�(��u��NQ�C�                                    Bx_2�2  �          @��׿�Q�?�\)��{�RC� ��Q�?������ffB�G�                                    Bx_2��  
�          @���Q�?�G���{�i33C���Q�?���=q�_{CB�                                    Bx_2�~  "          @�����L�����  C:�H��>�  ����
C+u�                                    Bx_2�$  T          @�p��)��>��H�}p��_�C)u��)��?aG��x���Y�C!�f                                    Bx_2��  "          @��ÿ!G�?���p�¡  C�׿!G�?�G���33W
B�                                     Bx_2�p  �          @�(��@  ?}p����B�B�{�@  ?�(������B�#�                                    Bx_3  "          @��\��p��
=���\CM�
��p��\)���
�C:B�                                    Bx_3�  �          @����8Q�?����z=q�Mz�CaH�8Q�?�=q�qG��C��C=q                                    Bx_3 b  �          @��׿�?����R�h��C�ÿ�@  ��  �X
=CB�                                    Bx_3/  o          @�녿�p�@z�����j�B��Ϳ�p�@/\)����V�B�W
                                    Bx_3=�  �          @�����=q@
=q���u�RB�#׿�=q@%��{�`�
B�B�                                    Bx_3LT  �          @�=q���@"�\���R�a�B�ff���@<���{��L�B�Q�                                    Bx_3Z�  
�          @��\�Q�@A��~{�N��B�.�Q�@Z=q�i���8p�B��                                    Bx_3i�  �          @�=q���
@�R����r��B��q���
@:=q��z��[�B���                                    Bx_3xF  �          @��׿@  @0  ����\�HB�k��@  @I���s�
�FffBγ3                                    Bx_3��  =          @�33��@{�hQ��E33B��\��@5��W��2Q�B�=q                                    Bx_3��  
?          @��H�N{=L���S33�6C3��N{>Ǯ�Q��533C-+�                                    Bx_3�8  �          @��\�y����=q�5��z�C7��y��<��
�5�=qC3��                                    Bx_3��  
          @�  �Q녿�(��9���ffCL.�Q녿��A��$(�CG��                                    Bx_3��  
q          @����J�H?�(��Dz��!��C�=�J�H@��8Q���Cp�                                    Bx_3�*  �          @�33�:�H@   �S�
�,=qC�\�:�H@��E�(�Cc�                                    Bx_3��  �          @��\�   ?�
=�i���D��CaH�   @�\�[��5Q�C	xR                                    Bx_3�v  9          @����  ?�(����z�B�8R��  ?�
=��(��~z�B�u�                                    Bx_3�  
�          @����e?�ff�8Q��*=qC ��e?�\)�z��	�C(�                                    Bx_4
�  �          @����O\)@0�׿��
���\C�{�O\)@:=q���R�~{C
!H                                    Bx_4h  
�          @�  �$z�?O\)�i���V
=C"�{�$z�?�
=�b�\�MffCJ=                                    Bx_4(  �          @����Q�?333��{�wG�C!� �Q�?�����H�mC�f                                    Bx_46�  "          @����%?����j=q�M�\C�=�%?�(��`  �A=qC^�                                    Bx_4EZ  
�          @���
=?�z��x���_��C�3�
=?����p  �S��CW
                                    Bx_4T   "          @������>�(���  �h�\C)�3���?\(��{��b\)C �                                     Bx_4b�  "          @��
�B�\?�{�%��RC\�B�\?�������\CJ=                                    Bx_4qL  
�          @�Q��Q�@3�
�^�R�6�\CY��Q�@8�ÿ���z�C
�
                                    Bx_4�  "          @�G��=p�@;����
����C\)�=p�@DzῙ���|  C��                                    Bx_4��  
�          @���"�\>��[��T33C)Ǯ�"�\?Q��W
=�N33C"
=                                    Bx_4�>  "          @���� �׿Q��z=q�_�HCF!H� �׾����~{�e�C<�R                                    Bx_4��  "          @�  �1녾����n{�S�C<B��1�    �p  �U�RC3�R                                    Bx_4��  �          @�
=�L�;8Q��Z�H�;33C7L��L��>B�\�Z�H�;33C0��                                    Bx_4�0  "          @�  �@�׾\�g
=�G�RC;+��@��<��
�hQ��I33C3��                                    Bx_4��  "          @�G��z�^�R��
=�w��CJ���z�����G���C?�                                    Bx_4�|  �          @�
=�,(��E��r�\�V�CC���,(���33�vff�[=qC;k�                                    Bx_4�"  �          @��\�z=q<#�
�
=q��C3�)�z=q>�  �	�����
C0c�                                    Bx_5�  
�          @�33���H>aG������z�C1{���H>�p���{��
=C/{                                    Bx_5n  
�          @������8Q��p���C6�=���=#�
���R�̸RC3��                                    Bx_5!  "          @�����u����(�C7G�����#�
��\�ӮC4!H                                    Bx_5/�  �          @��H����>���(���p�C2�����>k��#�
�G�C1!H                                    Bx_5>`  �          @�p����
>�(����R�qG�C.�q���
>�����J=qC.Y�                                    Bx_5M  
�          @�(����?@  �#�
��C*�=���?Q녿\)�޸RC)�3                                    Bx_5[�  T          @�z����;\��\)���
C8�����;aG���33��\)C6�H                                    Bx_5jR  "          @�p����
��  �޸R��Q�CD�����
�����\)��(�CB.                                    Bx_5x�  T          @������R��
=���H��ffCF�����R��(���{��  CD33                                    Bx_5��  �          @������R��(���Q���\)CD#����R�}p��z��̣�CA8R                                    Bx_5�D  "          @����s33���R����(�CIc��s33�����%����CE��                                    Bx_5��  "          @�p���
=���ͿУ����CB����
=�fff�޸R��{C@{                                    Bx_5��  	�          @��
��Q�333��=q��Q�C=\)��Q����33��C:�                                    Bx_5�6  "          @�=q��
=�(��z���G�C;Ǯ��
=�
=q�#�
�{C:��                                    Bx_5��  T          @��\����Y���\���C?�����L�;�����C>�)                                    Bx_5߂  "          @��R�hQ��*=q��  �xz�CX8R�hQ��\)��ff���CV��                                    Bx_5�(  "          @�(���z�8Q쾨����G�C=ٚ��z�.{��
=����C=B�                                    Bx_5��  
�          @�33��=q�:�H������HC=�\��=q�\)���H����C;c�                                    Bx_6t  f          @��\��=q�E������up�C>#���=q�!G������\)C<@                                     Bx_6  �          @�33��\)�W
=�u�@��C>�H��\)�O\)��{��\)C>5�                                    Bx_6(�  �          @�(������:�H>���@�  C=������E�>��R@x��C=��                                    Bx_67f  �          @�z����Ϳ�Q�?J=qA�CC
���Ϳ��\?&ffA ��CD�                                    Bx_6F  �          @��H���?�ff?Y��A+33C&c����?s33?xQ�AC�C'�q                                    Bx_6T�  �          @��H���
>�G�?�Q�Aq�C.:����
>���?�p�A{33C0�                                    Bx_6cX  �          @��
��  =#�
?s33A>{C3p���  ��\)?s33A=C4��                                    Bx_6q�            @��H��(�>�(�?���As�C.h���(�>�\)?��RA|Q�C0Q�                                    Bx_6��  T          @����G�?��
@33A�33C%����G�?G�@
=qA�\)C)�                                    Bx_6�J  �          @����(�>���?�ffA��C/����(�>\)?�=qA��\C2#�                                    Bx_6��  �          @��R���=�G�?�z�A�C2�=����u?�A�(�C4�q                                    Bx_6��  �          @�
=��Q�<��
?�\)A��C3����Q�\)?�\)A���C5�
                                    Bx_6�<  �          @���p�>�  ?�Q�A�Q�C0Ǯ��p�=�\)?��HA�Q�C3
                                    Bx_6��  �          @�\)��G�?8Q�?��
A�{C*ff��G�?   ?���A�{C-W
                                    Bx_6؈  �          @�p���G�?�G�@�A��
C%����G�?@  @(�A��
C)�=                                    Bx_6�.  �          @�(���?O\)?�G�A�p�C)���?
=?���A���C,
=                                    Bx_6��  �          @�(����?�?�\)A�(�C$B����?u?�  A�ffC&��                                    Bx_7z  �          @����p��?\(�@
=A���C'��p��?\)@��B�C+�=                                    Bx_7   �          @�\)�q�?L��@
=qA��C(
=�q�?�@  A��C,�                                    Bx_7!�  �          @�Q�����?��?��RA�=qC,� ����>�Q�?��A�z�C/�                                    Bx_70l  �          @�����?W
=?ǮA��RC((�����?#�
?�z�A�G�C*�R                                    Bx_7?  �          @�\)�s33?@  ?���A��C(�
�s33?�\?�33Aң�C,L�                                    Bx_7M�  �          @���u�?z�@A�33C+aH�u�>�\)@��A�C/�
                                    Bx_7\^  
�          @��R��Q�?0��@4z�B
�
C*(���Q�>���@8Q�B�\C/J=                                    Bx_7k  �          @�����\?�ff?�G�A��C%�����\?Y��?У�A�G�C(B�                                    Bx_7y�  �          @��R�U�?˅@�B��C���U�?��\@#33Bp�C(�                                    Bx_7�P  �          @����Q�@���ݙ�Cn�Q�@&ff��ff���C�)                                    Bx_7��  �          @��H�_\)?�?��A�p�C33�_\)?�p�?��A�
=C�
                                    Bx_7��  �          @���u@�\?uAB�\C=q�u@��?�  A~=qC�)                                    Bx_7�B  �          @����  ?�(�?W
=A*�RC����  ?�?��A^=qCG�                                    Bx_7��  �          @���q�@�
?��
AX  CaH�q�?�z�?�ffA�  C0�                                    Bx_7ю  �          @��R�j�H@.�R�W
=�%�Ck��j�H@3�
������\C�\                                    Bx_7�4  �          @��R�`  ?ٙ������  C��`  @   ����߮C=q                                    Bx_7��  T          @���z�?�Q������C!�)��z�?��R����P  C!J=                                    Bx_7��  �          @�  ��33?W
=��=q���HC)���33?�ff�������
C&xR                                    Bx_8&  �          @����W
=@  ��H��\)C8R�W
=@#33�ff��C��                                    Bx_8�  �          @����C�
@3�
�z���C	s3�C�
@E�����HC�                                    Bx_8)r  �          @�G��l(�?xQ��7
=�ffC%E�l(�?���,���
(�C�R                                    Bx_88  �          @�  �p  ?�Q��(���p�C"s3�p  ?��
�p���C�=                                    Bx_8F�  �          @�{����?.{����(�C*c�����?z�H�
�H�݅C&B�                                    Bx_8Ud  �          @�(��tz�aG����� Q�C78R�tz�=�G��=q� C2O\                                    Bx_8d
  �          @����c33?�G��8�����C u��c33?���,(��
�C.                                    Bx_8r�  �          @���XQ�?�z��2�\��RCs3�XQ�@��   ��=qC                                      Bx_8�V  �          @����L(�@33�,���
  C@ �L(�@)���
=��\)CT{                                    Bx_8��  �          @�Q��?\)@��7���C�H�?\)@#�
�"�\�p�CxR                                    Bx_8��  �          @�
=�P  ?���333�C���P  @��� ����CB�                                    Bx_8�H  �          @�G��Dz�@Q��;����CE�Dz�@ ���'
=�\)C��                                    Bx_8��  �          @����J�H?��
�S�
�0ffC  �J�H?�p��Fff�"�CaH                                    Bx_8ʔ  �          @�G��QG�?�\)�Fff�!�\C�H�QG�@�\�5��HC
=                                    Bx_8�:  �          @�
=�\��?�ff�)����C}q�\��@	������C\                                    Bx_8��  �          @�
=�K�?�ff�7���C�=�K�@��%��Q�C�=                                    Bx_8��  �          @�
=�S�
?����,���\)Ck��S�
@z������p�C�q                                    Bx_9,  �          @�
=�I��?�{�>{�z�Ck��I��@���+��
��CG�                                    Bx_9�  �          @���[�?�=q�Q���\)C޸�[�@����ģ�CO\                                    Bx_9"x  �          @����Fff?�Q��4z����Cn�Fff@z��#33�	�\C5�                                    Bx_91  �          @���Fff?�33�C�
�$ffC���Fff@z��2�\�ffC8R                                    Bx_9?�  �          @��@  ?��H�Q��233C��@  ?��B�\�!��C^�                                    Bx_9Nj  �          @�{�6ff?��H�`  �A�HC�3�6ff?��H�R�\�2C
                                    Bx_9]  �          @�ff�B�\?�Q쿙�����Cz��B�\@�h���]�Cs3                                    Bx_9k�  �          @�G��Y��@녿���(�C(��Y��@��B�\�*�HC��                                    Bx_9z\  �          @��H�Fff?\(��K��2�C$���Fff?����AG��'�\C                                    Bx_9�  
�          @��H�QG�?����@  �$p�C!�
�QG�?�  �3�
�{CT{                                    Bx_9��  T          @����J�H?Q��QG��4  C%� �J�H?���G
=�)G�C�\                                    Bx_9�N  �          @����W
=>��>{�$��C+�\�W
=?k��7
=��C$��                                    Bx_9��  �          @���Fff?��>{�-\)C*aH�Fff?u�7
=�%ffC"�=                                    Bx_9Ú  �          @�  �=p�?�  �;��%��C5��=p�?��*�H��\C�                                    Bx_9�@  �          @��\�G
=?޸R�s33�o33Cٚ�G
=?�{�.{�*=qC(�                                    Bx_9��  �          @�
=�E@&ff�!G��{C��E@*=q�k��Q�C8R                                    Bx_9�  �          @�=q�b�\?�{��=q��Q�CB��b�\@����
���\C�{                                    Bx_9�2  �          @���hQ�@ �׿�����(�C��hQ�@(����
�Z�\C�f                                    Bx_:�  �          @��\�h��?�
=���\����C)�h��@ff�u�MC\                                    Bx_:~  �          @���S�
@Q��  ��\)CE�S�
@%���\)�n�\C�                                    Bx_:*$  �          @����Z�H@ff�8Q��p�Cu��Z�H@(���33��G�C��                                    Bx_:8�  �          @�\)�C33@3�
�\��ffC	h��C33@5�=u?J=qC	(�                                    Bx_:Gp  T          @��\�C�
@*=q�\)����C�C�
@)��>�=q@w�C!H                                    Bx_:V  �          @��
�N{@"�\����z�C�q�N{@%��u�Y��CL�                                    Bx_:d�  �          @��\�W
=@�׾�����=qC
�W
=@�\�#�
���C��                                    Bx_:sb  �          @��\�J=q@   ��G��ƸRC�f�J=q@"�\�#�
�(�C:�                                    Bx_:�  �          @���AG�@2�\>#�
@�C	33�AG�@.�R?
=AQ�C	�\                                    Bx_:��  �          @���<��
@C33?333AO�
B�{<��
@9��?�A�(�B�                                    Bx_:�T  �          @�(�@�R@_\)?��
A��Be@�R@P  ?���A�  B^
=                                    Bx_:��  �          @��\?�
=@�z�?aG�A333B��H?�
=@|��?��
A���B��                                    Bx_:��  �          @�?��@��>�G�@�
=B�8R?��@�?�\)A_�
B�{                                    Bx_:�F  �          @�33?�
=@���>�ff@��RB�
=?�
=@{�?��A_�BQ�                                    Bx_:��  �          @�33?\(�@z�H���Ϳ���B��?\(�@xQ�?�@�33B��f                                    Bx_:�  �          @�  �.{@�z�>��@��Bƨ��.{@���?���Ak�B�33                                    Bx_:�8  T          @�=q�8Q�@��
��
=�q�B���8Q�@�Q�����B��                                    Bx_;�  T          @�Q��+�?�33@�\B�\C���+�?�  @$z�Bz�C�                                     Bx_;�  �          @���0��?�{@?\)B0��C���0��?\(�@K�B>��C"�3                                    Bx_;#*  �          @��R�"�\@   @(��BC��"�\?��@;�B233C��                                    Bx_;1�  �          @�G����
@�\@HQ�B<��B�����
?�  @]p�BX��C}q                                    Bx_;@v  �          @�Q쿠  @�@k�B[{B�p���  ?��@�  Bz�HB�k�                                    Bx_;O  �          @�z����@'�@,��B!
=B�Ǯ����@��@EB?{C��                                    Bx_;]�  �          @��\� ��@C33?�(�A�(�Cff� ��@.�R@��A��C��                                    Bx_;lh  �          @�p��)��@<(�?��HA�{C\�)��@,��?�
=A�C��                                    Bx_;{  �          @�  �!�@G
=?�Q�A��C{�!�@7�?�Q�A�z�C\)                                    Bx_;��  �          @��R�5�@AG�?E�A)p�C:��5�@5?��\A�  C��                                    Bx_;�Z  "          @�z��\��@�
?k�AN�HC(��\��?��?�  A�Cu�                                    Bx_;�   �          @�(��E@*�H>��@i��C��E@%?5A
=C                                      Bx_;��  �          @�z��:=q@4z�u�VffC�)�:=q@4z�>u@[�C�)                                    Bx_;�L  �          @���/\)@1녿^�R�J{C�f�/\)@8Q�Ǯ���C�\                                    Bx_;��  �          @����>�R@$z�Tz��>ffC=q�>�R@*�H�\��C
�                                    Bx_;�  �          @��
�h��?�(��aG��O
=C!���h��?��Ϳ(����RC��                                    Bx_;�>  �          @����aG�?�zΰ�
����C8R�aG�?�{���\�m�Cz�                                    Bx_;��  �          @����o\)?��\��
=���C$��o\)?�  ��p����HC!��                                    Bx_<�  �          @�p��G
=@&ff��p���ffC)�G
=@7
=��  ���C	h�                                    Bx_<0  �          @�{�X��?�\)�z����C}q�X��?�����G���p�C{                                    Bx_<*�  �          @��\�h��?�33� ����C"}q�h��?�������
=C��                                    Bx_<9|  �          @�
=�J�H?\�#�
�p�CO\�J�H?��������=qCk�                                    Bx_<H"  �          @����l(��z�����{C<޸�l(�����{�  C6E                                    Bx_<V�  T          @��\�tz�?8Q�����\)C)^��tz�?�{�����\C#�
                                    Bx_<en  
�          @�\)�z=q?�{�����C$��z=q?���\)��{C 
=                                    Bx_<t  �          @�\)�p��?�p���
��{C!�
�p��?�=q�����C:�                                    Bx_<��  �          @�p��o\)?�
=����G�C"n�o\)?��
�����p�C��                                    Bx_<�`  �          @�ff�z=q?&ff�   ��{C*���z=q?�  �������C%�                                    Bx_<�  �          @��~�R>��ÿ�33��\)C/=q�~�R?(�ÿ����C*��                                    Bx_<��  �          @�{�y��?�(���(�����C"���y��?�G���p����CǮ                                    Bx_<�R  �          @�ff�w�?��Ϳ�\)��\)C$��w�?��������C�
                                    Bx_<��  �          @�Q����
>�Q�����C/����
?+��ٙ���\)C*��                                    Bx_<ڞ  �          @�p��~�R?   ������C,�\�~�R?Tz���H����C(E                                    Bx_<�D  �          @��z=q?^�R��{���
C'u��z=q?�Q��
=��ffC#�                                    Bx_<��  �          @�{�n{?   ����HC,J=�n{?k������
=C&
                                    Bx_=�  �          @��X��?Tz��-p��=qC&0��X��?���� ���
z�C�=                                    Bx_=6  �          @�33�^�R?�\)��=qC"5��^�R?��
���Q�CQ�                                    Bx_=#�  T          @����tz�?�=q��\)��C$5��tz�?�{������
C c�                                    Bx_=2�  �          @��
�c�
?�p����
��33C�f�c�
@�R����s\)C�                                    Bx_=A(  �          @��\�Y��?�������33C:��Y��?�Q���R���HC��                                    Bx_=O�  �          @����c33?aG��\)���\C&{�c33?��
��\��p�C �                                    Bx_=^t  �          @����\��?B�\����C'}q�\��?����(���C ޸                                    Bx_=m  �          @����g
=?(��(�����C*n�g
=?�G���\��{C$c�                                    Bx_={�  �          @���_\)?0���=q���C(�_\)?���\)���C!��                                    Bx_=�f  �          @��H�n�R>��
�	����\C/�n�R?8Q��33��ffC)�                                    Bx_=�  �          @��H�j=q?+������z�C)���j=q?�������=qC#�                                    Bx_=��  �          @�(��`  ?����
=C!���`  ?˅�z����
C�=                                    Bx_=�X  �          @�z��Q�?�G��%��C��Q�?�p���
���HC:�                                    Bx_=��  �          @��R�Y��?^�R�.{�(�C%�H�Y��?����   �	G�C�R                                    Bx_=Ӥ  �          @����e?W
=�)����RC&޸�e?�=q�(���C��                                    Bx_=�J  T          @���dz�?=p��-p��Q�C(T{�dz�?�  � ���p�C �                                    Bx_=��  �          @�33�U?�\�6ff� ��C+J=�U?�ff�,���p�C"�                                     Bx_=��  �          @�p��Z�H?��H������Q�C+��Z�H@33���
���HC\                                    Bx_><  �          @����S33?�z��)�����CT{�S33@������Q�C                                    Bx_>�  �          @�Q��Mp�?�=q�G��$\)C�)�Mp�@	���0  ��C!H                                    Bx_>+�  �          @�(��B�\@
=�1����CL��B�\@'
=��
��C\)                                    Bx_>:.  �          @���:�H?�ff�)����\Ch��:�H@��\)���C�                                    Bx_>H�  �          @�  �E�?���   ��RC��E�?�{�
�H����C��                                    Bx_>Wz  �          @y���^{?�p��\(��L��C�q�^{?�\)����{C�                                    Bx_>f   �          @����U@
�H�
=��HC��U@\)����ffC+�                                    Bx_>t�  �          @vff�E@�R����u�C=q�E@�R>B�\@:�HC+�                                    Bx_>�l  �          @j�H�/\)@ff>��@��CW
�/\)@p�?n{Al��C
                                    Bx_>�  �          @�Q��U�?����\����C}q�U�@�
��{�ř�CL�                                    Bx_>��  T          @����e?��� ������C8R�e@  �˅��Q�C�
                                    Bx_>�^  �          @���`  ?�{�   ��C5��`  @�Q��܏\C(�                                    Bx_>�  �          @���N�R@�=q� =qC��N�R@"�\��
=���
C�\                                    Bx_>̪  �          @�\)�Q�?�(��G���p�C��Q�@�ÿ�����p�C�                                    Bx_>�P  �          @�G��]p�@ff��
=��ffC�q�]p�@��<��
>aG�C=q                                    Bx_>��  �          @��H�h��?�ff>L��@0��C�f�h��?�p�?\)@�p�C�f                                    Bx_>��  �          @�(��dz�?�?O\)A6=qC�dz�?У�?�A���Cn                                    Bx_?B  �          @��R�c�
?��H?
=A33C&f�c�
?�ff?z�HA\Q�C33                                    Bx_?�  �          @\)�j�H?\=#�
?z�C���j�H?�p�>�33@�33C�                                    Bx_?$�  �          @����33?��׾�p���{C!ff��33?�
=���
��ffC ��                                    Bx_?34  �          @���Q�?�33��������C$���Q�?�Q콸Q쿏\)C$c�                                    Bx_?A�  �          @��H��{?��ͽ�G���Q�C%:���{?���=�?��C%@                                     Bx_?P�  �          @�=q��p�?��=#�
>��HC%k���p�?�ff>��@`��C%�{                                    Bx_?_&  
�          @�=q���?�\)>#�
@z�C$޸���?���>Ǯ@�p�C%�)                                    Bx_?m�  �          @�����
?�      <��
C##����
?�(�>��@aG�C#�                                     Bx_?|r  �          @������\?��=�\)?p��C"@ ���\?�G�>�33@��
C"��                                    Bx_?�  �          @�(����?Q녾�������C)����?\(����У�C(xR                                    Bx_?��  �          @�����?�?.{A�C ����?�  ?xQ�AM�C"�                                    Bx_?�d  �          @�Q��}p�?�=q?J=qA+�C!z��}p�?���?��Af=qC$)                                    Bx_?�
  �          @������?c�
�&ff��
C(�)���?�G���ff��(�C'!H                                    Bx_?Ű  T          @�{����?333�}p��C\)C+33����?fff�Q�� ��C(�=                                    Bx_?�V  �          @�{���?B�\�J=q��RC*�=���?h�ÿ�����C(�3                                    Bx_?��  �          @�����\?   �5��C-����\?#�
�
=���C,�                                    Bx_?�  �          @����33?\)����C-���33?(�þǮ����C+�=                                    Bx_@ H  �          @����G�?&ff�����C+����G�?=p�������=qC*�3                                    Bx_@�  T          @�z����\?#�
��Q���
=C,
=���\?333�W
=�(Q�C+B�                                    Bx_@�  T          @�
=��z�?L�;��
�x��C*=q��z�?Y���\)��C)��                                    Bx_@,:  �          @��R��(�?J=q��{���
C*L���(�?Y���#�
��33C)��                                    Bx_@:�  �          @������?Y��=�\)?^�RC)Y�����?Q�>��@K�C)�=                                    Bx_@I�  �          @���33?O\)>W
=@"�\C*���33?=p�>\@�p�C*��                                    Bx_@X,  �          @�����>��    =L��C.=q����>�=�G�?���C.ff                                    Bx_@f�  �          @��R��{>���aG��*�HC0�
��{>����#�
��33C0Q�                                    Bx_@ux  �          @�Q���      �\)��  C4���  <��\)���HC3��                                    Bx_@�  �          @�Q���  >#�
�u�+�C2���  >.{���
�W
=C1�                                    Bx_@��  �          @�����Q�=�Q�aG��%C2޸��Q�>\)�B�\�G�C2Q�                                    Bx_@�j  �          @�Q����>W
=�����\)C1n���>�\)��=q�S�
C0��                                    Bx_@�  �          @����ff>��k��4z�C.8R��ff?�����Q�C-�q                                    Bx_@��  �          @�\)��?������G�C,����?&ff��G����C,&f                                    Bx_@�\  �          @�  ���?\(��\���\C)�����?k��.{�ffC(�=                                    Bx_@�  T          @�����z�?z�H�z���Q�C(���z�?����{��(�C&�=                                    Bx_@�  �          @�(�����?�Q�?5Ap�C$������?}p�?uA@z�C'J=                                    Bx_@�N  �          @����l��?�(�?�\A�G�C!�l��?G�@   A���C({                                    Bx_A�  �          @�ff�aG�?�\)?�\Aȣ�C��aG�?n{@�\A�C%G�                                    Bx_A�  �          @���e?�=q?˅A���C(��e?�?�A�\)C!޸                                    Bx_A%@  �          @����o\)?�{?��A���C ��o\)?u?�A���C%�\                                    Bx_A3�  �          @����p��?��?�Q�A�33C#Ǯ�p��?+�?�33A�C)�H                                    Bx_AB�  �          @�ff�e?�z�?���AиRC"��e?333@z�A�C(�3                                    Bx_AQ2  �          @�p��e�?��H?�33Ȁ\C���e�?�(�@  A��C!:�                                    Bx_A_�  �          @�Q��c�
@�\?��A��C:��c�
?��@�
A���C��                                    Bx_An~  �          @��
�Vff@��?�ffA���Cc��Vff?�z�@  A��\C�H                                    Bx_A}$  �          @��R�E�@�R?�ffA�ffC��E�?�  @G�B�
Cff                                    Bx_A��  T          @�33��33?z�?���A�\)C+�f��33>�\)?��A��HC0�                                    Bx_A�p  �          @�G���(�<#�
?���An=qC3ٚ��(���  ?���Ag�
C7u�                                    Bx_A�  �          @�����
>Ǯ?���A�C.�����
=�\)?���A��
C3{                                    Bx_A��  �          @��\�k�?�\)?���AƏ\C�)�k�?fff@A�
=C&Q�                                    Bx_A�b  �          @�\)�s33?G�?�{A�z�C(p��s33>�p�?�  A�
=C.p�                                    Bx_A�  �          @�33�w�?��\?�
=A��\C%@ �w�?
=?��A��C+ff                                    Bx_A�  �          @���o\)?�?�Q�A��C+��o\)?z�H?�p�A��C%c�                                    Bx_A�T  �          @���w�?�ff?�z�A�p�C!z��w�?\(�?�
=A�p�C'��                                    Bx_B �  �          @�����33?�G�?�A�(�C&8R��33?�?�{A�\)C,
=                                    Bx_B�  �          @�G���33?��
?��A��C%޸��33?��?�A�C+�H                                    Bx_BF  �          @�
=����?�z�?˅A�=qC#�����?=p�?���A�(�C)��                                    Bx_B,�  
�          @����y��?��
?���A�Q�C!Ǯ�y��?s33?�p�A���C&E                                    Bx_B;�  �          @����qG�?�{?Y��A5C��qG�?���?��\A�=qC{                                    Bx_BJ8  �          @�{��=q?O\)?
=@�z�C)ff��=q?#�
?E�A   C+�{                                    Bx_BX�  �          @��R��\)?��=��
?z�HC!�
��\)?���>��@�33C"��                                    Bx_Bg�  �          @�  ����?�  >���@�Q�C#�����?���?!G�AC%.                                    Bx_Bv*  �          @�{�e�?Ǯ?�=qA��\Cs3�e�?�
=?�
=A��C!��                                    Bx_B��  �          @���o\)@�\?&ffA
�RCQ��o\)?���?��At��C
=                                    Bx_B�v  �          @���w�@�
�fff�3�C��w�@p���z��h��C�                                    Bx_B�  �          @�  �p  @�\��  �L��C�{�p  @{�Ǯ����C��                                    Bx_B��  �          @��R�QG�@(���p���=qCO\�QG�@2�\�����_\)C��                                    Bx_B�h  �          @�ff�q�@�
�����C�\�q�@\)?��@�CL�                                    Bx_B�  �          @�ff�c33@p�>u@S33C(��c33@�?J=qA-p�C�R                                    Bx_Bܴ  �          @���]p�?��R?�(�A�z�C)�]p�?�ff?�
=A��
C�
                                    Bx_B�Z  T          @���hQ�@G�>�p�@��C�3�hQ�@ff?n{AH(�C�H                                    Bx_B�   �          @�G��n{@�?��@�Q�C��n{@ff?�At��C�                                    Bx_C�  �          @�\)�@  @,��?�p�A��C
  �@  @
=@&ffBz�C�f                                    Bx_CL  �          @�  �9��@(��@�A�C	���9��@ ��@.�RB�
CE                                    Bx_C%�  �          @xQ�n{@ ��@.�RB5�B��Ϳn{?�p�@Q�Bl
=B�                                    Bx_C4�  �          @�(��\(�@33@\��Be33B�\)�\(�?��@w
=B�z�C �                                    Bx_CC>  �          @�33���@*=q@:�HB6Q�B�G����?���@`��Bl�B�8R                                    Bx_CQ�  �          @���\(�@.{@AG�B:��B���\(�?���@g�Brp�B�R                                    Bx_C`�  �          @��R��@1G�@G
=B?Q�Bɀ ��?�\)@n{By�RB�aH                                    Bx_Co0  �          @�
=�W
=@7�@AG�B9�B�� �W
=?��R@j=qBu\)B�=q                                    Bx_C}�  �          @��׿��\@>{@\)B
�HB�8R���\@�\@;�BA{B�                                      Bx_C�|  �          @����G
=@P��?��Ab�HC�f�G
=@6ff?���A�=qC	z�                                    Bx_C�"  �          @��H�L��@H��?uA@��C}q�L��@1�?޸RA��HC
�3                                    Bx_C��  T          @�p��W
=@E?z�HAA�C	p��W
=@.{?�  A�=qC��                                    Bx_C�n  �          @��
�\��@;�?n{A9�C���\��@%�?�A�33C5�                                    Bx_C�  �          @�=q�^�R@3�
?p��A=�C
�^�R@p�?�33A��HC��                                    Bx_Cպ  �          @�=q�\��@4z�?z�HAF=qC��\��@p�?�Q�A��C�                                    Bx_C�`  �          @�G��|(�@+��u�:�HC���|(�@&ff?&ff@��HC��                                    Bx_C�  �          @��H�`  @9��>�z�@l��CW
�`  @.{?�ffAU�C&f                                    Bx_D�  �          @�z��\(�@*�H>�
=@�\)C0��\(�@p�?�{Ak�
Ch�                                    Bx_DR  �          @��
�L��@0  ������
CJ=�L��@333>W
=@333C
��                                    Bx_D�  �          @���K�@:=q�u�L��C	u��K�@7�?��@��C	޸                                    Bx_D-�  �          @��\�_\)@7
==L��?0��C���_\)@/\)?Q�A'�C�\                                    Bx_D<D  �          @�=q�W
=@E�������C	aH�W
=@W����ٙ�C�                                    Bx_DJ�  �          @�  ���@<���)���ffC\���@c�
��  ��  B��                                    Bx_DY�  �          @�  ��Q�@@  �O\)�,�B����Q�@qG���
�陚B�B�                                    Bx_Dh6  �          @��H��p�@�H�Y���H��B��´p�@P  �'����B�
=                                    Bx_Dv�  �          @�33���@Vff�   ��p�B�(����@qG����\�Mp�B��                                    Bx_D��  "          @�=q�
=q@j=q��������B�
=�
=q@}p��
=q�ۅB�33                                    Bx_D�(  �          @��Ϳ�
=@�G���Q��n{B����
=@��R�#�
��B�G�                                    Bx_D��  �          @�33�@  @!G��ff��G�C�3�@  @Dz�������\C^�                                    Bx_D�t  �          @��H�A�@-p��$z��z�C
0��A�@S�
��p���
=Cs3                                    Bx_D�  �          @�ff�G
=@'
=��=q��Q�C���G
=@AG�����X��C޸                                    Bx_D��  �          @��\�S�
@33��\)��
=CB��S�
@=q�}p��Z�\C��                                    Bx_D�f  �          @��\�[��Ǯ�333��C:}q�[�>����333�ffC-W
                                    Bx_D�  �          @�33�Tz�>�G��+���RC,��Tz�?�z��p���C �R                                    Bx_D��  �          @�33�~�R?�����z�����C#(��~�R?�Q��  ���RC
=                                    Bx_E	X  �          @�  �l��?���33���C��l��@�Ϳ���b�RCG�                                    Bx_E�  �          @����W�?�G�������G�Ch��W�@Q쿂�\�dz�C�R                                    Bx_E&�  �          @��
����>��R����=qC/������?0�׿��
��=qC*L�                                    Bx_E5J  �          @��
�J=q@����z���  C�J=q@1G��k��FffC
�q                                    Bx_EC�  T          @���e�@�\��ff��(�CY��e�@z�+���
C
=                                    Bx_ER�  �          @����c33?�׿���mp�C.�c33@ff���H���Cn                                    Bx_Ea<  �          @��H�a�@(����H����C(��a�@(��
=q���CL�                                    Bx_Eo�  �          @��Q�@,�Ϳ����}�C�\�Q�@;��Ǯ���\C
=q                                    Bx_E~�  "          @�  �Y��@1G�����U�C�)�Y��@<�;W
=�*�HC�                                    Bx_E�.  �          @�
=�s33?�(��.{�ffC���s33?ٙ�>���@�
=C�f                                    Bx_E��  �          @�Q��xQ�@�p���@(�C���xQ�@G������vffC��                                    Bx_E�z  T          @����h��@\)��Q����HC�\�h��@333�+����Cc�                                    Bx_E�   �          @��
�l��@녿\��z�CG��l��@(Q�L��� ��C��                                    Bx_E��  �          @�=q�u@{������
=C@ �u@1녿.{��RC{                                    Bx_E�l  �          @����xQ�@C33�h���&=qC�=�xQ�@L(�<#�
>\)C�)                                    Bx_E�  �          @��R�{�@9���Tz��C���{�@@��=#�
?�\C�=                                    Bx_E�  �          @����h��@8Q�z�H�?33C���h��@B�\��G����C(�                                    Bx_F^  �          @����P��@5��{��33C
���P��@L(��8Q���HC�f                                    Bx_F  �          @��Tz�@1녿ٙ���
=C��Tz�@J=q�Q��!G�CxR                                    Bx_F�  �          @�{�W
=@+���\)��Ck��W
=@G
=��  �F{C	:�                                    Bx_F.P  �          @�(��|��@=q��p���(�C���|��@/\)�5�33C:�                                    Bx_F<�  �          @���(�@%��s33�)��C� ��(�@/\)�#�
��(�C�                                    Bx_FK�  �          @�p�����@0�׿=p��33CO\����@6ff=���?�z�Cn                                    Bx_FZB  �          @��R��33@=p���Q��Tz�C#���33@K���=q�?\)C+�                                    Bx_Fh�  �          @��\�xQ�>�@�\A�=qC,��xQ�\)@A�=qC6{                                    Bx_Fw�  �          @�\)��\)?�z�?.{@��C$5���\)?��?���A<��C'\                                    Bx_F�4  �          @�\)��ff?�G�?5@��\C#  ��ff?��R?���AG\)C&                                      Bx_F��  �          @�p����R?�Q�?L��A�C&�\���R?c�
?�\)AHz�C)�=                                    Bx_F��  �          @��
��Q�?fff?�\@�C)Ǯ��Q�?5?@  A33C+��                                    Bx_F�&  "          @�����=q?L��>��@���C+
=��=q?�R?0��@�ffC-�                                    Bx_F��  �          @�(���G�?@  >��H@�
=C+����G�?�?0��@�  C-�                                    Bx_F�r  �          @�(�����?z�H>�p�@��\C)  ����?Q�?#�
@�p�C*�3                                    Bx_F�  �          @����\)?���?E�A	G�C'����\)?Q�?��A>ffC*�f                                    Bx_F�  �          @�z����R?��?:�HA�HC'�
���R?Q�?��
A8Q�C*�
                                    Bx_F�d  �          @��H��
=?Tz�?.{@�\)C*�=��
=?z�?fffA#\)C-L�                                    Bx_G

  �          @����G�?
=q?�@�{C-���G�>�33?+�@�  C0�                                    Bx_G�  �          @�ff��{>.{=���?��C2���{>�>��?�{C2�                                    Bx_G'V  �          @�����ff?(��>�@��RC,�R��ff>��H?(��@��C.��                                    Bx_G5�  �          @�  ��?��?�@��C-�3��>�Q�?.{@�p�C0                                    Bx_GD�  �          @�����{>�p�?333@�p�C/޸��{>\)?G�A��C2h�                                    Bx_GSH  �          @������#�
?333@�Q�C;(�����Tz�>��@��RC=.                                    Bx_Ga�  �          @�{��논�?}p�A(��C4L���녾���?n{A33C7�=                                    Bx_Gp�  �          @�z����>u?\(�A�
C1k���녽#�
?fffAG�C4ff                                    Bx_G:  �          @����Q���?�G�A-C5����Q��ff?k�Ap�C8�H                                    Bx_G��  �          @��H����=L��?aG�A�C3h����׾aG�?Y��A
=C6aH                                    Bx_G��  �          @������    ?Y��A
=C3�q���þ��?O\)A�
C6�
                                    Bx_G�,  �          @��
���׾�
=?n{A   C8�����׿0��?B�\A��C;n                                    Bx_G��  �          @�(��������?&ff@�\)C:xR�����E�>�G�@�z�C<O\                                    Bx_G�x  �          @����H�aG�?W
=A�HC6Q����H���?:�H@���C8�q                                    Bx_G�  �          @����=q���?\(�A
=C9��=q�5?.{@�C;�H                                    Bx_G��  �          @����=q��(�?(��@��C8����=q��R>��H@�Q�C:�H                                    Bx_G�j  �          @�p����
�
=q���
�W
=C9�q���
���H�u�%C933                                    Bx_H  �          @��\��=q�.{����z�C5�H��=q�#�
���
�\(�C5��                                    Bx_H�  �          @�{���Ϳ�\�B�\� ��C9Y����;�
=��{�eC8u�                                    Bx_H \  �          @�G���Q�#�
=�G�?���C5����Q�B�\=L��?��C5�                                    Bx_H/  �          @�����R��(�>W
=@(�C8u����R���=u?.{C8�                                    Bx_H=�  �          @�����(���p�=u?#�
C7�H��(���p��u��RC7��                                    Bx_HLN  �          @�����ͽ�G�<#�
=��
C5����ͽ��ͼ���\)C5
                                    Bx_HZ�  �          @�\)��
=�L��=u?(��C6\��
=�W
=    ��C6(�                                    Bx_Hi�  �          @�  ���R�u>��H@���C6z����R�\>Ǯ@�33C7�q                                    Bx_Hx@  �          @����=q�Q�?��
A}�C<����=q���\?�Q�ADQ�CAff                                    Bx_H��  �          @�(���(����?�  Ax(�C9ٚ��(���  ?�  AMp�C>�=                                    Bx_H��  �          @��\���R��=q?�=qA3�C6�=���R���?p��A�
C:@                                     Bx_H�2  �          @������8Q�?�  A&�\C5�)�����H?c�
AQ�C9�                                    Bx_H��  �          @������R���
?uA�RC4޸���R�\?aG�A�C8�                                    Bx_H�~  �          @�������<#�
?���AH(�C3޸���;�p�?��A>{C7�3                                    Bx_H�$  �          @�����>�Q�?��Ahz�C0(������G�?�
=Ao�C5&f                                    Bx_H��  �          @����Q�#�
?�=qA1��C4ff��Q�\?�G�A%�C8                                      Bx_H�p  �          @�33����.{?���A/�
C5� ����   ?uAG�C9:�                                    Bx_H�  �          @����G�>��?���Ac\)C.�R��G�<#�
?�Ap��C3�                                    Bx_I
�  �          @�z���ff?=p�?�=qA9��C+�H��ff>�33?�G�AYp�C0!H                                    Bx_Ib  �          @�  ��{>�
=?
=@�
=C/����{>L��?333@��C1�)                                    Bx_I(  �          @��H��=q    >aG�@G�C4��=q��\)>W
=@��C4                                    Bx_I6�  �          @����
=>�\)>�
=@�(�C1���
=>�>��H@��C2�H                                    Bx_IET  �          @�G����׾aG��L���33C6E���׾\)����+�C5z�                                    Bx_IS�  �          @�33���\��\)�����B�\C6�����\�.{��p��tz�C5�q                                    Bx_Ib�  �          @�z���z����G���33C5:���zὣ�
������RC4�=                                    Bx_IqF  �          @�=q��녾u��\)�6ffC6� ��녾\)��{�`��C5k�                                    Bx_I�  �          @����33�L�;����Tz�C6���33���
��p��tz�C4�)                                    Bx_I��  �          @�  ����k������HQ�C6ff�������Q��p  C5:�                                    Bx_I�8  �          @�\)��
=�W
==L��?�\C6&f��
=�W
=���
�k�C633                                    Bx_I��  �          @�������k������\C6T{�����.{�B�\��(�C5Ǯ                                    Bx_I��  �          @�=q��녾��R��G�����C7&f��녾�  �L���
=C6��                                    Bx_I�*  �          @�����(����
����C7B���(�����aG��(�C6��                                    Bx_I��  �          @�p���(���Q�����G�C7�H��(��8Q�\)��
=C5�{                                    Bx_I�v  �          @��R��ff�.{��  �   C5����ff��\)��z��:=qC4��                                    Bx_I�  �          @�  ��  �#�
�8Q����C5����  ��Q�k��G�C4�                                    Bx_J�  �          @��������k��C5J=����#�
����(��C4n                                    Bx_Jh  �          @������׼�����=qC4O\����<��
�\)����C3ٚ                                    Bx_J!  �          @���������\)��Q�Y��C4�������#�
���Ϳ��C4aH                                    Bx_J/�  �          @��
��=q<��#�
����C3���=q>u�
=����C1�                                     Bx_J>Z  �          @�33���=L�Ϳ�{�aC3s3���?�\��G��Qp�C.�f                                    Bx_JM   @          @��
����L��=�\)?5C5������W
=    <��
C6�                                    Bx_J[�  �          @�(���(��8Q켣�
�.{C5�=��(��.{���
�E�C5��                                    Bx_JjL  �          @�ff��{�u<�>���C6E��{�k��L�;�C6@                                     Bx_Jx�  �          @�����;L��>\)?�\)C5�R���;u=u?�RC6Q�                                    Bx_J��  �          @��R��p���;�  ��C9B���p���(���
=��=qC8!H                                    Bx_J�>  �          @�����\)�\)>8Q�?�
=C9Y���\)�
=���
�uC9��                                    Bx_J��  �          @�����R�z�        C9�����R�
=q�L�Ϳ��C98R                                    Bx_J��  �          @���������>\@h��C7�f�����\>aG�@C8��                                    Bx_J�0  �          @��R��p��\?�@���C7�3��p��
=q>�p�@dz�C9=q                                    Bx_J��  �          @�{��p���>�\)@*=qC5:���p��W
=>aG�@�C6�                                    Bx_J�|  �          @�(���=q�0�׽��
�O\)C:����=q��R���
�EC:�                                    Bx_J�"  T          @��
���H��
=�.{��
=C8����H��{���R�@  C7L�                                    Bx_J��  �          @�����\)��\�@  ��C9���\)�aG��aG��\)C6+�                                    Bx_Kn  �          @�����Q쾸Q�aG��(�C7����Q쾊=q�����S33C6��                                    Bx_K  �          @�Q���  ��=q����  C6�3��  ��  ����C6s3                                    Bx_K(�  �          @��
��33���;�\)�-p�C7����33��\)�����z=qC6Ǯ                                    Bx_K7`  �          @�{���
���ͿG���C7�H���
���Ϳ^�R��RC5�                                    Bx_KF  �          @��
��녾u�@  ���
C6h����<��J=q��  C3��                                    Bx_KT�  �          @��H���þ��ÿ8Q���=qC7=q���ýu�J=q��Q�C4��                                    Bx_KcR  �          @�(���=q��ff�333�ٙ�C8p���=q�8Q�O\)��p�C5Ǯ                                    Bx_Kq�  �          @���������O\)��\)C58R����>.{�O\)����C2Y�                                    Bx_K��  �          @�ff��(�=��
�aG��(�C3:���(�>\�L����ffC0J=                                    Bx_K�D  �          @���������G���RC9� ����z�p���  C6                                    Bx_K��  �          @�Q���ff��=q�=p��ᙚC6����ff<#�
�G���  C3�f                                    Bx_K��  �          @�p����R>�
=��Q��a�C/����R?fff�����;�
C+�                                    Bx_K�6  T          @��R��
=?�=q�����RC(W
��
=?�(��\�w\)C!�                                     Bx_K��  �          @��R��p�?!G���Q����C-T{��p�?�Q쿮{�\z�C'��                                    Bx_K؂  �          @�z���G�=#�
��  �#\)C3�
��G�>Ǯ�k��ffC/�q                                    Bx_K�(  �          @����G��\)�aG����C5h���G�>8Q�^�R�33C2�                                    Bx_K��  �          @�  ��33<��
����F�\C3�{��33>�׿����8��C/ff                                    Bx_Lt  �          @�����
�B�\��z��3\)C5޸���
>k���z��2=qC1��                                    Bx_L  �          @�  ���;�=q���� (�C6�)����=������$z�C2ٚ                                    Bx_L!�  �          @�(����׾��s33�C8�����׽�Q쿆ff�%p�C4�)                                    Bx_L0f  �          @�Q���33��  �h�����C>���33�(������@��C::�                                    Bx_L?  �          @�ff��
=���H��G��#�C@n��
=�E���\)�_\)C;��                                    Bx_LM�  �          @�\)���\�Tz�s33�Q�C<�����\��G���
=�>�RC8�=                                    Bx_L\X  �          @�
=���;�\)�Y���Q�C6�H����=#�
�c�
��C3��                                    Bx_Lj�  �          @����{��\���\�!�C9{��{��G������4(�C5{                                    Bx_Ly�  �          @����p����:�H��33C9:���p��aG��^�R��
C6=q                                    Bx_L�J  T          @�\)�������.{���C7����#�
�=p���ffC4h�                                    Bx_L��  �          @�Q���
=���z�����C5Q���
==�Q�
=���
C3&f                                    Bx_L��  �          @�  ��\)�#�
��G���z�C4#���\)>����
=��z�C2��                                    Bx_L�<  �          @�ff��{����\)�6ffC533��{�#�
���R�EC4
                                    Bx_L��  �          @�33���\����Ǯ��  C6�����\���;����RC5{                                    Bx_Lш  �          @��
��33���þ����B�\C7ff��33�L�;������C6�                                    Bx_L�.  �          @�(����<���p��qG�C3�R���>#�
��{�Z�HC2^�                                    Bx_L��  �          @�����<�����C3������>L�;�
=���C1��                                    Bx_L�z  �          @��\���þ���+���C5������=��Ϳ.{��Q�C2�                                    Bx_M   �          @�����������(�����C5����<��
������C3�                                    Bx_M�  �          @�����Q�W
=�8Q��z�C6:���Q����  �%C5aH                                    Bx_M)l  �          @�\)��\)�B�\��G���33C6
=��\)�\)�.{��C5z�                                    Bx_M8  �          @�Q���
=��
=�����33C8h���
=�aG��
=���
C6T{                                    Bx_MF�  �          @��R��{�.{�\��G�C5Ǯ��{���
��
=���C4.                                    Bx_MU^  �          @�  ��
=���
��ff���C4޸��
==�Q��ff��
=C3�                                    Bx_Md  �          @�{��p�=�\)���H��{C38R��p�>�  ��(���G�C1^�                                    Bx_Mr�  �          @��
��33=���
=��Q�C2����33>�=q��33�n�RC1#�                                    Bx_M�P  �          @�����Q�=#�
�����N{C3����Q�>\)����6ffC2p�                                    Bx_M��  �          @�(����
=���    �L��C2����
=�Q�<�>�33C2�q                                    Bx_M��  �          @�{��p�=����
�a�C2����p�>k���  �333C1�                                     Bx_M�B  �          @�\)��ff>��
��
=���\C0}q��ff>�ff��=q�>{C/�                                    Bx_M��  �          @�
=��ff>�  �������C1B���ff>\��z��L��C/Ǯ                                    Bx_Mʎ  �          @��R��{>u��p����HC1^���{>�Q쾅��5�C0
=                                    Bx_M�4  �          @������H>��
=��(�C.�����H?+���p���p�C,�=                                    Bx_M��  �          @�z����?�R�!G��߮C-  ���?O\)��33�z=qC*��                                    Bx_M��  �          @����=q?(��333���HC-)��=q?Tz��(���\)C*�3                                    Bx_N&  �          @�������?(��G���C-)����?\(��   ��G�C*^�                                    Bx_N�  �          @����z�?W
=�(���p�C*����z�?�G���  �/\)C(�H                                    Bx_N"r  T          @��
��\)?�G��(�����C)���\)?��B�\�   C'h�                                    Bx_N1  �          @�����>�����  �*�HC0�q����?&ff�O\)�
=C,��                                    Bx_N?�  �          @����\=�\)�}p��(  C3J=���\>�(��c�
��C/c�                                    Bx_NNd  �          @��R��zἣ�
�Y����
C4:���z�>����L����\C0��                                    Bx_N]
  "          @����{�\)�!G�����C5}q��{=�G��#�
��
=C2�H                                    Bx_Nk�  �          @��R��ff�u��{�c�
C4�{��ff=��
�����aG�C333                                    Bx_NzV  �          @�����zᾊ=q�u��RC6���z�k��#�
��C6xR                                    Bx_N��  �          @�p�������
=L��>��HC4�������Q�<#�
>��C4�                                    Bx_N��  �          @��
����L��<��
>k�C4������L��    <��
C4�\                                    Bx_N�H  
�          @�����\?�G�>��?�33C&{���\?�{?(�@׮C'�                                    Bx_N��  �          @��H���
?�p�<��
>B�\C#)���
?�{?�@�
=C$\)                                    Bx_NÔ  �          @�
=���\?�{>�=q@G
=C'
=���\?n{?+�@�  C)!H                                    Bx_N�:  
�          @�z����R?��8Q��G�C&�����R?�33>�\)@G�C&�f                                    Bx_N��  "          @�������?z�H�����A��C)  ����?��ÿ&ff���HC%J=                                    Bx_N�  "          @��
��33?���#�
��{C'���33?�����(���=qC                                      Bx_N�,  �          @�p���G�?�
=��\)�E�C!����G�?�=q?�@�C"��                                    Bx_O�  "          @�33��  ?�\)��
=����C c���  ?��>��
@UC 8R                                    Bx_Ox  �          @�33����?���=���?���C �)����?�
=?L��A  C"\)                                    Bx_O*  
�          @��H���?Ǯ�u� ��C#�q���?��>�p�@r�\C#��                                    Bx_O8�  T          @��\���
?�p���=q�5�C$�����
?�(�>��R@L��C$��                                    Bx_OGj  
Z          @�  ����?�
=���
�Q�C$�����?�Q�>�  @(��C$��                                    Bx_OV  
�          @�  ��G�?�(��Ǯ���\C$xR��G�?�G�>L��@�C$
                                    Bx_Od�  �          @�����?�\)���R�L(�C%�=���?���>u@!G�C%h�                                    Bx_Os\  
�          @��\����?��;��
�S33C%������?�\)>aG�@G�C%                                    Bx_O�  
�          @�G����?�=q�������
C&{���?���>\)?��HC%��                                    Bx_O��  
�          @�����?��\���H��\)C&^����?�{<�>�{C%k�                                    Bx_O�N  �          @�{����?�p���Q��qG�C&�
����?��
>\)?\C&h�                                    Bx_O��  �          @��
���R?��
��=q�:�HC&:����R?��
>u@#33C&(�                                    Bx_O��  "          @�z���\)?�  ��=q�8Q�C&�
��\)?�G�>k�@��C&��                                    Bx_O�@  "          @��\����?��\>u@%�C&&f����?�=q?5@�C((�                                    Bx_O��  
�          @����G�?���>�p�@w�C#���G�?��?uA z�C%��                                    Bx_O�  �          @����
=@�����
=CO\��
=@�R>��
@VffC�{                                    Bx_O�2  	�          @��
��\)@����33��{C)��\)@6ff�
=��(�C�                                    Bx_P�  
	          @�z���Q�?��H��
=��  C!(���Q�@z῏\)�6�RC#�                                    Bx_P~  
�          @�����ff@녿��
��p�C����ff@#�
�\(���
C��                                    Bx_P#$  
�          @����?�ff��Q���Q�C����@=q����4  C�3                                    Bx_P1�  �          @�ff��?&ff��z����HC,����?��Ϳ�G��{�
C%p�                                    Bx_P@p  �          @�����Ϳk��{����C>�����=�G��(Q���=qC2��                                    Bx_PO  �          @�\)��?�\�1���ffC-����?Ǯ�
=�̏\C!��                                    Bx_P]�  	`          @�����R?=p��P  ���C*
=���R?����,����C5�                                    Bx_Plb  
�          @�  ����?\)�O\)�C,������?��
�1G���(�C��                                    Bx_P{  T          @������?��Fff�C,�\���?�p��(Q���ffCQ�                                    Bx_P��  "          @�p���?
=�@  ��RC,s3��?��H�"�\����Cٚ                                    Bx_P�T  T          @�\)�e�=����
�C��C2��e�?�(��o\)�-(�CJ=                                    Bx_P��  "          @�=q�j�H>��R����A�HC/=q�j�H?�z��mp��'z�C�=                                    Bx_P��  
Z          @�Q����>����|(��-Q�C/ff���?����_\)���C�                                    Bx_P�F  
�          @��H����?���33��33C8R����@ �׿}p��&=qCk�                                    Bx_P��  
�          @�=q��=q@�\��
=�p  C�{��=q@*�H��
=���\C�                                    Bx_P�  �          @����Q�?��Ϳ���3�C���Q�@���  �%C�                                    Bx_P�8  
�          @��
���
?�Q��\����C$L����
@G�����.=qC}q                                    Bx_P��  
Z          @������?}p������Q�C(������?��Ϳ�����C��                                    Bx_Q�  "          @�p���G�?�z��0����ffC&Y���G�@
=q�����C�                                     Bx_Q*  T          @�{��z�?Q��^{�=qC)u���z�@��7
=��p�CB�                                    Bx_Q*�  �          @������?���.{��ffC'{����@���
��G�Cu�                                    Bx_Q9v  T          @�����\)?�\)�   ��33C$����\)@33��ff�X��C��                                    Bx_QH  T          @�  ��(�?��H���qG�C&�3��(�?�
=�Tz��33C!��                                    Bx_QV�  �          @�Q�����?�G���  �RffC).����?��@  ���C$��                                    Bx_Qeh  
�          @�����p�?(�ÿY���C-\��p�?p�׿�����C*!H                                    Bx_Qt  �          @�����Q�>�녿�ff���HC/����Q�?}p���G��T  C)^�                                    Bx_Q��  
�          @�33��  ?
=��ǮC-O\��  ?�(������C#�f                                    Bx_Q�Z  
�          @�z�����>\�9����G�C/ff����?�  � ������C"�=                                    Bx_Q�   T          @�33��  >�Q��:=q��C/���  ?��R�!���C"��                                    Bx_Q��  �          @�=q���R>�Q��9����ffC/�
���R?��R�!G���{C"}q                                    Bx_Q�L  A          @�{���\?B�\�N�R�ffC*�����\?�p��*�H����C��                                    Bx_Q��  5          @�Q���  >��J�H�33C.���  ?�
=�.{��
=C ��                                    Bx_Qژ  T          @�������?�Q��i���p�C$u�����@#33�7��CG�                                    Bx_Q�>  
�          @������?�R��{���C-L����?��ÿ��H�o33C&�                                    Bx_Q��  
�          @�(�����?G���  ��G�C+�����?�����L(�C%�)                                    Bx_R�  �          @�������?z�H�8Q���G�C(�q����@33�  ���\C��                                    Bx_R0  �          @�p�����?��
�(Q���Q�C##�����@�Ϳ�ff���\C\                                    Bx_R#�  T          @�p����@33�"�\��  Ch����@HQ쿴z��\��C�3                                    Bx_R2|  �          @����
=@,���&ff���
C�{��
=@aG���ff�NffC�=                                    Bx_RA"  	�          @�\)��p�@G��@  ��Cc���p�@C�
��Q���
=CQ�                                    Bx_RO�  
;          @�Q��e?c�
����A�\C&&f�e@�R�]p����C^�                                    Bx_R^n  �          @��R�~�R?G��s�
�,�RC(޸�~�R@\)�J�H��C�
                                    Bx_Rm  �          @�p���G�?+��dz����C+
��G�@33�@  � ��Cz�                                    Bx_R{�  "          @��
��p�?�(��Q���(�C� ��p�@!녿�p��yC��                                    Bx_R�`  T          @��H����?fff�l���'�C'p�����@�
�A��{C=q                                    Bx_R�  �          @���s33?���33�;��C,8R�s33@��b�\���C��                                    Bx_R��  
�          @���w�?��u��1�HC,@ �w�@G��S33�33C^�                                    Bx_R�R  
�          @�33�g�>����u��9�
C/O\�g�?��XQ��ffCk�                                    Bx_R��  �          @��\�8��?��\�
=�.�HC ���8��?����Q�C�H                                    Bx_RӞ  "          @�{�'
=@U�?�A��C ��'
=@�
@@  B"�HC
n                                    Bx_R�D  �          @�p�� ��?�(��
=�ffC}q� ��@�H���R���C��                                    Bx_R��  
�          @�
=>�p��L����p�ª�HC��>�p�?���G�ǮB�Q�                                    Bx_R��  �          @�G�>#�
�8Q���
=£��C�^�>#�
?�������B��                                    Bx_S6  "          @��=��
������\©��C��R=��
?�p����Q�B���                                    Bx_S�  "          @�p�>�Q�����z�«C���>�Q�@�\����)B��                                    Bx_S+�  "          @�z�?n{�����  �HC��)?n{?�p���  \Bw33                                    Bx_S:(  "          @�  ?O\)��{��z�¡�C���?O\)?�\)��=q�B�L�                                    Bx_SH�  �          @��?O\)=u��\)£33@�G�?O\)@G����\)B��f                                    Bx_SWt  �          @�Q�?��H>�{���H�Ayp�?��H@%���
=�p�
B��3                                    Bx_Sf  
Z          @���?G�=�\)��ff¤ff@�  ?G�@Q���z�B��q                                    Bx_St�  T          @�?n{��z����\ 8RC�c�?n{@   ��
=�B�8R                                    Bx_S�f  �          @��?c�
��\)��G�¡��C���?c�
@����HB�B��                                    Bx_S�  �          @�@/\)?�{��33�o��A��@/\)@E���R�6=qBA�                                    Bx_S��  
�          @�z�?��?xQ���z��{Bz�?��@K���Q��W��B�Ǯ                                    Bx_S�X  
�          @�33?��
?��\���u�B��?��
@N{���R�V��B��q                                    Bx_S��  T          @�
=?\?�(���Q�{B0G�?\@_\)��ff�?G�B���                                    Bx_S̤  �          @�z�>��H?�p���z���B�k�>��H@c33��=q�Iz�B�Q�                                    Bx_S�J  
�          @�
=�L��?8Q����
¤ǮB�LͽL��@5����i\)B�{                                    Bx_S��  �          @�=q��{���
��33«ǮC@���{?�z����RB�33                                    Bx_S��  �          @�ff�   @�Q�>k�@*�HB��)�   @���@33A�Q�B��f                                    Bx_T<  �          @���33@�{���
��Q�B��)��33@|��?�A��B�aH                                    Bx_T�  �          @�녿�\)@�p�?�\@�(�B�LͿ�\)@fff@
�HA�\)Bݞ�                                    Bx_T$�  �          @��׿�z�@U�@G�B&  B�LͿ�z�?�G�@��RB}�B��                                    Bx_T3.  T          @�33����@O\)@���BC��B�(�����?��@�  B�ǮC��                                    Bx_TA�  �          @�����
@\)@\(�BO�B�33���
?\(�@��B�.CT{                                    Bx_TPz  �          @��ÿ��@,(�@.�RB,\)B�𤿑�?�=q@fffB�C��                                    Bx_T_   �          @�����
@HQ�@�B G�B�����
?��H@QG�BS�B���                                    Bx_Tm�  "          @�{�   @�p�?.{A (�B䞸�   @p  @�A�ffB�
=                                    Bx_T|l  �          @���   @�녿����B��   @���>��@�
=B�Ǯ                                    Bx_T�  �          @�{�  @s33����p�B�8R�  @j=q?��An�\B�8R                                    Bx_T��  T          @�ff�'
=@��@�Q�BG��C�H�'
=>��R@���Bo�C-O\                                    Bx_T�^  �          @�=q�.�R@�\@|��B?�
C��.�R>��@�G�Bj�C*@                                     Bx_T�  �          @��R�,(�@0��@_\)B(z�CG��,(�?��@�=qB`=qC\                                    Bx_TŪ  T          @�����R@Dz�@��B\)B�����R?�@\��BH��C��                                    Bx_T�P  �          @����\@=p�@p�B�RB�Ǯ��\?��@N�RBPz�C�R                                    Bx_T��  T          @qG���G�@33@.�RB;�
B�Ǯ��G�?@  @UB|��C�3                                    Bx_T�  
�          @�\)��  @`��>��
@�  B�B���  @Dz�?�p�A�\)B�{                                    Bx_U B  
�          @�{>B�\@0��@]p�BMffB��>B�\?��@���B�
=B��R                                    Bx_U�  �          @�  ��\)@XQ�@/\)Bp�BظR��\)?���@x��Bo�B���                                    Bx_U�  T          @��ÿ��H@hQ�?ٙ�A�
=B�q���H@'
=@B�\B,{B��H                                    Bx_U,4  �          @�(���@AG��
=q��\)C �=��@>�R?@  A0��C5�                                    Bx_U:�  �          @�Q쿥�@&ff�\)�"�\B��Ϳ��@Y���������
Bݞ�                                    Bx_UI�  �          @��R��@Q�@.�RB0p�C�
��?O\)@W
=Blp�C&f                                    Bx_UX&  	�          @�ff�B�\@]p�?�p�A�(�B��H�B�\@(�@@  BF�B�Ǯ                                    Bx_Uf�  T          @-p�?@  ?������  B�H?@  ?�\)>��@��HB�B�                                    Bx_Uur  
Z          @vff�	��?
=@A�BV��C$���	���5@@  BS�CFaH                                    Bx_U�            @^{���?�(�@(Q�BH  C
{���>��@@  Bt33C+(�                                    Bx_U��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_U�d              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_U�
              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_U��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_U�V              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_U��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_U�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_U�H              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_V�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_V�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_V%:              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_V3�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_VB�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_VQ,              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx_V_�  ,          @��?���?xQ���=q��B33?���@Y�����
�X��B��H                                    Bx_Vnx  �          @�Q�?��?k����H��B��?��@W
=����]B���                                    Bx_V}  
�          @�=q?�=q?�G����HL�B��?�=q@\������X�B��                                    Bx_V��  
�          @ʏ\?��?���\��B=q?��@_\)���H�V�HB���                                    Bx_V�j  �          @ə�?�{?�33��=q� B8?�{@dz���G��UB��{                                    Bx_V�  �          @ə�?\(�?�G���33�
B^��?\(�@l(������T  B���                                    Bx_V��  �          @ȣ�?��?�\)��Q�z�BM�
?��@p  �����M��B�z�                                    Bx_V�\  "          @ʏ\?�=q?�33���R��BL�\?�=q@~�R��\)�A�RB�                                    Bx_V�  "          @˅?�p�?������=BD��?�p�@s�
����KQ�B�W
                                    Bx_V�  �          @˅?��?�33�����HB<  ?��@r�\��p��KG�B���                                    Bx_V�N  
�          @˅?xQ�?�ff�����BU  ?xQ�@o\)�����Rp�B���                                    Bx_W �  
�          @��?�(�?�(�������B[  ?�(�@��\�����AQ�B��
                                    Bx_W�  �          @�?B�\?��
���
��B�?B�\@�����\�C
=B�W
                                    Bx_W@  "          @�
=?��?ٙ��ƸR�B�33?��@�(���{�GG�B��                                    Bx_W,�  �          @�
=?@  ?��H��  �\Bz?@  @{����\�O
=B�L�                                    Bx_W;�  "          @�=q?��R?�{��\){B*(�?��R@u����H�LQ�B���                                    Bx_WJ2  �          @Ӆ?˅?�����W
B
=?˅@e�����UB�W
                                    Bx_WX�  �          @�33?�?!G���{��A�\)?�@P�����H�i
=B�Q�                                    Bx_Wg~  "          @�=q?�z�?=p���p��qB�R?�z�@Vff�����e33B�Ǯ                                    Bx_Wv$  "          @ҏ\?��\?}p����
��B  ?��\@c�
��(��[�RB��R                                    Bx_W��            @��?�=q?
=q�θR  A�z�?�=q@L(������j=qB��
                                    Bx_W�p  
l          @�  ?��
�����\)��C��f?��
@{��  �|ffBX��                                    Bx_W�  
�          @�G�?h��>���ָR¤L�AQ�?h��@;���G��|G�B�W
                                    Bx_W��  
(          @�  ?�z�(����G�C�"�?�z�@��\B?�H                                    Bx_W�b  
�          @�z�@R�\��Q���=q�d��C�4{@R�\?��\����j��A�p�                                    Bx_W�            @�p�?�33�\)��p��C���?�33@
�H��=q��BS(�                                    Bx_Wܮ  
          @�G�?�z�>\)���H@���?�z�@7����v�Bz�                                    Bx_W�T  �          @׮@333��p���p��z��C�(�@333?��������x��A��                                    Bx_W��  �          @أ�@^{�������e��C�t{@^{?�33����a(�A��                                    Bx_X�  
Z          @�=q@Z�H�z�H�����iz�C���@Z�H?\����b�A�\)                                    Bx_XF  �          @���@X�ÿ����
�j\)C�t{@X��?�����=q�gp�A�                                    Bx_X%�  �          @ۅ@p���{��ffC���@p�@Q���Q��v\)B<�                                    Bx_X4�  �          @ۅ@.{�\(���{z�C�@ @.{?�ff����u�B��                                    Bx_XC8  �          @�33@,(�?aG���ff�)A���@,(�@X�������J\)BN                                      Bx_XQ�  "          @׮@
�H?��R��\)B�A�  @
�H@n�R�����H{Bo�                                    Bx_X`�  B          @�\)?�Q�?xQ����H��A�R?�Q�@a�����Wz�B�                                    Bx_Xo*  
�          @�(�?s33>���
=£@�33?s33@:�H�����|p�B��f                                    Bx_X}�  "          @���@{?����θR��Bz�@{@�(���33�5�RBrff                                    Bx_X�v  "          @�ff@\)?�(��ə�A�ff@\)@}p����
�>33Bg�\                                    Bx_X�  �          @ڏ\@��>k��ʏ\u�@�p�@��@6ff����b��BG��                                    Bx_X��  �          @��
?��
?��R�ҏ\.B]�R?��
@�33����N�
B��
                                    Bx_X�h  
�          @ۅ>\@ ������=qB�W
>\@�����33�@��B�W
                                    Bx_X�  T          @�=q��  @���z�B�W
��  @������\�5(�B�                                    Bx_Xմ  �          @��ÿk�?��R��p���B�Ǯ�k�@��\��\)�@��BʸR                                    Bx_X�Z  
Z          @���@�\��Q�k�B�k���@������R�733B�Q�                                    Bx_X�   �          @߮��G�@%����Q�B�Ǯ��G�@�����  �+�\B�                                    Bx_Y�  �          @߮�Y��@333�ʏ\p�B�녿Y��@����H�$�\BƮ                                    Bx_YL  "          @���B�\@,������B�k��B�\@��\����'��B���                                    Bx_Y�  T          @����=q@7��ȣ���B��q��=q@�\)��Q��"�B��f                                    Bx_Y-�  
�          @�ff����@P  ��(��w��B�.����@�Q���
=��B�aH                                    Bx_Y<>  
�          @����Q�@J=q���
�z(�B��ᾸQ�@�p�������B��{                                    Bx_YJ�  �          @�33��@XQ���p��o��B�.��@�G��~�R�{B�                                    Bx_YY�  �          @�33�}p�@_\)����g�BӸR�}p�@�33�tz���RB��                                    Bx_Yh0  "          @�=q����@(���p��3B�p�����@�G����H�)p�B�8R                                    Bx_Yv�  �          @�녿���@C33��(��qz�B䞸����@�
=��=q�(�Bъ=                                    Bx_Y�|  �          @�ff�=p�@�  ��{�NQ�B��H�=p�@�(��J�H�ٙ�B��
                                    Bx_Y�"  �          @�ff�Tz�@�z���=q�H
=Bɀ �Tz�@ƸR�@���ͅB�G�                                    Bx_Y��  
�          @��H�Q�@�33��\)�;Q�B�LͿQ�@�Q��'
=���
B�                                      Bx_Y�n  
�          @׮�B�\@x����
=�Z=qB�� �B�\@���Vff��B��                                    Bx_Y�  T          @�G�>�@��R���\�N�\B�ff>�@����E��Q�B��                                     Bx_Yκ  "          @ۅ?k�@������/  B��?k�@�z������B��3                                    Bx_Y�`  �          @��?�R@��H���R�$  B�8R?�R@Ϯ��Q���Q�B�(�                                    Bx_Y�  
�          @أ�>���@��
���H��B��=>���@��H���R�K\)B��q                                    Bx_Y��  "          @ָR�5@�33�fff��B�ff�5@��H�p�����B�B�                                    Bx_Z	R  �          @�z�0��@����8���ϙ�B�Q�0��@�=q�.{��G�B���                                    Bx_Z�  �          @�녿W
=@��y����\B�Q�W
=@��H����D��B�                                    Bx_Z&�  �          @���(�@�z���Q��6��B�
=��(�@�
=�   ���RBϮ                                    Bx_Z5D  
�          @����@w
=��33�H
=B����@�(��A�����B�33                                    Bx_ZC�  
�          @�����
@b�\���H�U�B�ff���
@�{�XQ�����B�=q                                    Bx_ZR�  �          @�z��\@;�����k=qB�W
��\@����{���RB��                                    Bx_Za6  �          @�(��  @�����H�.��B��)�  @�Q���H���\Bޙ�                                    Bx_Zo�  �          @�\)�%@�z���z��.ffB�
=�%@�{� ������B��)                                    Bx_Z~�  
�          @�=q�z�@���l(����B��z�@�  �����9�B�=q                                    Bx_Z�(  �          @�33�=q@��\����%p�B�B��=q@�  �
�H����B�                                    Bx_Z��  �          @�  �5@����8���ԏ\B����5@��׿
=q����B�k�                                    Bx_Z�t  �          @љ��4z�@�{�333���HB���4z�@�z�Ǯ�\(�B��                                    Bx_Z�  T          @�33�'
=@�33�dz����B잸�'
=@��Ϳ�p��,��B���                                    