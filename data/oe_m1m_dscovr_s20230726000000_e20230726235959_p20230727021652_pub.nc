CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230726000000_e20230726235959_p20230727021652_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-07-27T02:16:52.128Z   date_calibration_data_updated         2023-05-09T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-07-26T00:00:00.000Z   time_coverage_end         2023-07-26T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx���   �          A��
=����@��HA�z�C�׿�
=��
=@�33A�ffC�                                    Bx����  �          A
ff�0�����@���B��Cv���0�����@�{B(�Cu�                                    Bx��
L  �          Ap��s�
�g�@ʏ\BIG�C_��s�
�K�@��BS��C[ٚ                                    Bx���  T          @��H@�Q쿕�.�R��HC��H@�Q쿬���)��� 33C��\                                    Bx��'�  T          @�@�ff?�p���{� {AK�
@�ff?h�������#�A�                                    Bx��6>  
�          A@�(��������>Q�C�Q�@�(������(��9{C�U�                                    Bx��D�  T          AG�@��I����{�Cp�C�]q@��g���ff�:��C���                                    Bx��S�  T          A=q@W
=��(����ffC�  @W
=��\)��ff��(�C�w
                                    Bx��b0  
�          A�@QG���ff�5����C�8R@QG���(������\C���                                    Bx��p�  T          A�H@]p���=q�vff�Џ\C�@]p���=q�Vff��C��H                                    Bx��|  �          @�Q�@����G���Q��=G�C�\@�����
�z�H� z�C��3                                    Bx���"  �          A�@�����
��{�*ffC�Ф@����
=��{�陚C���                                    Bx����  �          A33@��\��
=����� Q�C�q@��\��녿�����C��3                                    Bx���n  �          A�@��������
�Mp�C�5�@�����׿\��RC�                                      Bx���  �          A@�\)��ff�J�H��{C��{@�\)����)����Q�C��
                                    Bx��Ⱥ  �          @�p�@G
=���ÿ����RC���@G
=��33�!G���{C�Ф                                    Bx���`  �          @�
=�&ff��  >���@h��C�׿&ff��ff?W
=@��RC���                                    Bx���  �          @ۅ��=q��=q��Q��f=qC��\��=q�����H�$z�C��{                                    Bx����  
�          @�33?����녿��k�C�~�?����p����H�)C�t{                                    Bx��R  �          @�ff�aG���z�����z�C���aG����׿����n�RC��
                                    Bx���  �          @�33�Q���G�>�Q�@0  Ct0��Q��׮?Y��@ϮCt
=                                    Bx�� �  o          AG������Ϳ�����z�Cn�=����
=�#�
���\Cn�                                    Bx��/D  T          A�����\�!G�@��
B	\)C8�)���\�u@���B
G�C5��                                    Bx��=�            A Q��p���{@��B�
C;�\�p��.{@�(�B�C8��                                    Bx��L�  
�          Az�� (���33@��RBffCAQ�� (���G�@��HB�C>��                                    Bx��[6  
�          A=q�����&ff@�
=A��HCF� �������@���A�
=CD=q                                    Bx��i�  
�          A��z����@7
=A�
=C@^��z����@>�RA��C>�                                    Bx��x�  
Z          A=q��
=��G�@|��A�{C@����
=���H@�=qA�
=C>��                                    Bx���(  
�          A (����>���@���B
=C1�=���?B�\@ÅB�HC.s3                                    Bx����  
�          A����p�?��@��B!(�C/^���p�?���@�  B=qC+�                                    Bx���t  �          A�
����?�p�@�
=B	�C+!H����?��@��Bp�C(@                                     Bx���  �          A��녿���@j=qA���C>5��녿���@qG�A�z�C<p�                                    Bx����  �          A�
�R��\)@eA�Q�C@+��
�R�˅@n{A�33C>ff                                    Bx���f  T          AQ��p��W
=@���A�(�CJ�=�p��AG�@���A�
=CHz�                                    Bx���  �          AG������\@XQ�A��CP������@mp�A��CN�
                                    Bx����  
�          A�������G�@B�\A�G�CQ�{�����G�@XQ�A��\CPu�                                    Bx���X  �          A����=q����@�p�B�C_n��=q���@��B�HC]
                                    Bx��
�  �          A���������@��B=qC\�)�������@\B)�CY��                                    Bx���  T          Az����\��G�@��RBG�C\�����\���
@�=qB��CZ
=                                    Bx��(J  �          A�R�s33��z�@}p�A��Cq)�s33���@��A�33Co��                                    Bx��6�  �          Aff��G�����@�G�A�z�Ci��G�����@�G�B��Ch)                                    Bx��E�  
�          A�������H@s33A�ffCkE�����Q�@��\A��Ci��                                    Bx��T<  
u          A����z���
=@.�RA���Cj����z���\)@R�\A�Q�Ci�f                                    Bx��b�  T          A����G���{?���A��ClJ=��G��ᙚ?���AM��Ck�=                                    Bx��q�  �          A��^{��?��AK�Cuu��^{��{@\)A�33Ct�3                                    Bx���.  �          Az��|(����H>�
=@.�RCsL��|(�����?�ff@�33Cs!H                                    Bx����  �          A{��33��\�-p�����Cp��33����
=�\��Cqff                                    Bx���z  
�          A�
��{��ff��=q�"ffCm����{�񙚿xQ��ƸRCn&f                                    Bx���   
�          Ap�������\=�Q�?\)Cln������G�?5@��RClO\                                    Bx����  �          A���333��  ��{�=qCz&f�333���H�8Q���
=CzY�                                    Bx���l  T          A
=q��(���Q��{���z�C�c׿�(�����S33��33C��\                                    Bx���  
�          A�ÿ333����~�R��{C�W
�333��=q�U��Q�C�p�                                    Bx���  �          A �þ������<����33C�uþ�����������33C��H                                    Bx���^  
�          AQ��%���  ?�z�AH��Cy���%����H@\)A�Q�CyJ=                                    Bx��  "          A�k������HQ����HC�Ff�k�����!����RC�e                                    Bx���  �          A �ÿ�Q��\)?J=qAl��Ck�
��Q��=q?�  A�(�Ck                                    Bx��!P  
�          A  �7��P  @���BJ�\Cd�=�7��2�\@�G�BX��C`33                                    Bx��/�  T          A33�J=q�ٙ�@h��A���Cu
�J=q��
=@�ffA�(�Cs�R                                    Bx��>�  �          A���<(���  �\)���Cvk��<(������33�FffCv�                                    Bx��MB  k          A�\�����\)�33�z�\C�Ф�����(����,��C��f                                    Bx��[�  �          A�?O\)�����I����z�C��?O\)�����\)��G�C���                                    Bx��j�  �          A�
�=p��{���vffC�w
�=p��=q>#�
?��C�y�                                    Bx��y4  �          Az��G��������/�C�l;�G���H�\(���Q�C�q�                                    Bx����  
(          A녾�  �g�@���Bg��C��)��  �Fff@�=qB{��C��f                                    Bx����  "          A녿@  ����A
=B�Q�C;���@  ?.{A�\B���C	�f                                    Bx���&  "          A
=�����p�A=qBd�RCr������|(�AQ�BvffCn��                                    Bx����  �          A%G���p����H@��B(�CpaH��p���=q@��
B��Cn��                                    Bx���r  T          A+\)������@���A�
=Cm��������@���A�33Clc�                                    Bx���  T          A0�����p�@;�Aup�Cq������@p  A�ffCp8R                                    Bx��߾  T          A!�������
=@2�\A�(�CX�������{@R�\A�ffCWaH                                    Bx���d  T          A�R��  ��  @AG�
C[����  ��G�@'�A|(�CZ�)                                    Bx���
  �          A(�������{?��@O\)Cf�R����� ��?�=q@�  Cfk�                                    Bx���  T          A2{������?O\)@�Cc�������
=?��@�\)Cb�H                                    Bx��V  T          A333����=q��\)���Ci.���������4z�Cis3                                    Bx��(�  �          AG���(������33�333Cm����(���
��33�׮Cn{                                    Bx��7�  �          AG��{���=q�u����CsL��{��=q�Fff���Ct8R                                    Bx��FH  T          A��u��   �p�����Ctk��u����@  ��33CuE                                    Bx��T�  
�          A��L(���33��  ���Cw�f�L(��\)�`  ��p�Cx�=                                    Bx��c�  	�          A���+���p���p��p�C���+�������{C���                                    Bx��r:  Q          Aff@�(��$z���Q��>{C�:�@�(��J�H�Ǯ�3�C��q                                    Bx����  �          AG�>�\)���H��{�Qz�C�y�>�\)��\)����;�
C�Q�                                    Bx����  �          A���{��
=��  ��C�����{��\)�����(�C���                                    Bx���,  �          A�?�����
��z��R�C�N?����  ���
�=�\C��                                     Bx����  �          A
ff@z��~{��  �f\)C�(�@z��������H�S
=C�Ff                                    Bx���x  
�          A�
?�G�������{�9��C��f?�G����
���\�$(�C�Z�                                    Bx���  "          A=#�
���H����\)C�'�=#�
����ff�33C�%                                    Bx����  �          A�������ff��
=��RC�e������\)��  �z�C��H                                    Bx���j  �          A{��Q���  �Å�$ffC�����Q���������
=C��                                    Bx���  T          A{?#�
��  ��\)�3{C��f?#�
�ڏ\���H��C���                                    Bx���  T          A���33��������  C����33�Q������33C��                                    Bx��\  "          AG������p�������C��������p���z�����C�*=                                    Bx��"  �          A���p�������G��C�|)��p���������C��
                                    Bx��0�  �          A
�H�8Q���Q���ff�-p�C��R�8Q��ڏ\������C��q                                    Bx��?N  �          A
ff��������7
=C�
=�����Q������ Q�C�|)                                    Bx��M�  �          A�ÿ�{�����G��(�C}���{�߮����=qC
                                    Bx��\�  �          A{������G��=qCy�����{����Q�C{^�                                    Bx��k@  �          A������p�����6G�C��ÿ����������
=C���                                    Bx��y�  �          A��o\)��G���z��IffCc33�o\)��ff��ff�7�\Cgu�                                    Bx����  �          A��?�ff��(�?ǮATQ�C��H?�ff��{@p�A�p�C�)                                    Bx���2  �          A\)�8Q���ff��33�133C���8Q��������R��C���                                    Bx����  �          A�R�O\)��(���  ��HC�g��O\)��z�����G�C���                                    Bx���~  �          A=q?\)��33����=qC��=?\)��=q��p����HC�]q                                    Bx���$  �          A ��?�33�����Å�G(�C��\?�33�����=q�0��C��\                                    Bx����  �          A��>���z���ff�^\)C�%>��������R�F\)C��q                                    Bx���p  T          @��R��H��{��p��A  Cr�q��H��G���z��+33Cup�                                    Bx���  �          A�?k��������
�S��C�N?k����\����<  C���                                    Bx����  T          A	�@>{���������K  C�&f@>{�����6��C�7
                                    Bx��b  �          A���	녿�\)�B�\��  C@8R�	녿��<#�
=L��C@J=                                    Bx��  �          Aff�
=����L�;�33C?�)�
=���
>\)?c�
C?�{                                    Bx��)�  �          A	���z��R�\���J�HCKQ���z��Tz���Ϳ333CK�                                    Bx��8T  �          A�
���H�h�ÿ������CM�H���H�q녿�ff���CNu�                                    Bx��F�  �          @�����������A���Q�C[�)�������R�!����C]�                                     Bx��U�  �          @��
���\��ff��
=�\)Cg�����\��(��hQ��܏\Cj{                                    Bx��dF  �          A���j�H��(������Cl���j�H���H�u����Cn�                                    Bx��r�  �          A������G��mp��مCV  �����p��O\)��=qCXu�                                    Bx����  T          A
=��(�����Q��D=qC�p���(���ff�����+(�C�
=                                    Bx���8  �          A�H�J=q��  ���H�E\)C�c׿J=q������33�+�C��=                                    Bx����  �          Az�u�Ϯ��\)�?=qC��\�u��Q��θR�%�RC�@                                     Bx����  �          A z�.{�������
�G=qC��3�.{��
=��33�-�C�J=                                    Bx���*  �          A  �
=q�����Q��E�C�j=�
=q��  ��G��+��C��{                                    Bx����  �          A%��(�� z���H�Z�HCi���(��  ���R�=qCj�                                    Bx���v  T          A$Q�������`�����RCk������Q��%��j{Cl�{                                    Bx���  �          A%������p��dz�����Cn�q����
=�%�f�RCo�=                                    Bx����  "          A����p����H�r�\����Cm���p���\)�8Q����HCnW
                                    Bx��h  �          AQ����\������z���\)Cd�����\�Ϯ�Z=q���HCf�{                                    Bx��  �          A���
��G��K���Q�CZ�H���
��(��"�\��Q�C\Y�                                    Bx��"�  T          Az���{��ff����b=qCY�
��{��p���G���CZ�R                                    Bx��1Z  �          A{��G�����XQ����Cm!H��G��ff�(��d��Cn=q                                    Bx��@   �          A Q��u��H�����  Ctٚ�u�
{�Tz����RCv�                                    Bx��N�  "          A  �%��33��z���  C{0��%����l����Q�C|E                                    Bx��]L  �          A	G���=q��ff�Z�H���\Cn����=q�ᙚ�&ff��p�Cp                                      Bx��k�  �          A	p��8Q���G�������Q�Cw  �8Q���Q��[����CxQ�                                    Bx��z�  �          A�R��p���
=��  ���C�����p���
=�l(���G�C��                                    Bx���>  �          @��R��{��ff��33��\C|����{���fff��  C}�{                                    Bx����  �          A�R@j�H��z�����3p�C��@j�H��{��z��\)C�]q                                    Bx����  �          Ap���33��G��8������Cf����33���H�33�L  Cg��                                    Bx���0  �          A�������  �{����Cfz�������H�����ChY�                                    Bx����  
�          A����H�ָR�
=q�Z�RCd�����H��p��������Ce޸                                    Bx���|  �          A�\��  � ���\)�T��Cn#���  �(���p���Q�Cn�\                                    Bx���"  �          A,Q쿀  �
=�ƸR��C�����  �ff���
��RC��H                                    Bx����  �          A�^{������H��{Cv��^{����u���p�Cwn                                    Bx���n  �          A�
�g
=��\�������HCs� �g
=�����Fff��
=Cu{                                    Bx�   T          Aff�W���{���
��RCvY��W��\)�g����Cw�3                                    Bx� �  
�          A#33�7
=������p����Cy}q�7
=�p�������\C{�                                    Bx� *`  �          A%G��_\)���H�������Ct�{�_\)�G�������z�Cv��                                    Bx� 9  �          A*{��\)��ff�ƸR��Cpk���\)��H���R���Cr��                                    Bx� G�  T          A+���{����ff�G�Ch#���{��ff��G����Ck�                                    Bx� VR  �          A   �w
=��p����\����CsJ=�w
=�(�������p�Ct�3                                    Bx� d�  �          A$���S33�
�R��������Cy!H�S33���8������Cz{                                    Bx� s�  �          A"ff�8Q��
=���JffC��H�8Q��=q�h����
=C���                                    Bx� �D  �          A"�H�.{����-p��~{C�y��.{��ÿ����=qC�}q                                    Bx� ��  �          A%G��B�\�G��G���  C��R�B�\��\���(  C�˅                                    Bx� ��  T          A.�\����#��K���=qC�G�����(�Ϳ���C�N                                    Bx� �6  �          A0  �y��� �Ϳ���  CxǮ�y���"ff���
��
=Cx�R                                    Bx� ��  �          A((���{���}p���  C����{�Q��.{�{�
C�5�                                    Bx� ˂  T          A.{@(����w
=���C�y�@(�����'
=�mp�C�&f                                    Bx� �(  |          A0��@�ff��z��tz���z�C��@�ff���\�333�k�C�H�                                    Bx� ��  �          A:�R@�33���H�tz���Q�C��q@�33� z��1G��]p�C�,�                                    Bx� �t  �          A;33@�����n{���HC���@���
�   �E�C���                                    Bx�  "          A.�H@љ��=q��
�/33C�  @љ��	p��fff����C���                                    Bx��  �          A-��A
=�׮�z��0z�C�FfA
=�޸R������Q�C���                                    Bx�#f  �          A4Q�A�R���H�R�\��z�C�ФA�R����#�
�S\)C��                                    Bx�2  T          A=��A:=q=�Q쿳33��33>�A:=q���Ϳ�33��33C��H                                    Bx�@�  �          A:{A3\)?�?}p�@��RA�A3\)@�\?5@e�A$��                                    Bx�OX  T          A@  A)�@XQ�@��HA�(�A�
=A)�@|(�@u�A�\)A���                                    Bx�]�  �          A>�RAp�@��@�
=A�\)A�z�Ap�@�{@p  A��A�                                    Bx�l�  �          A4Q�A(�@�  @���A�ffA�A(�@��H@s�
A��B(�                                    Bx�{J  h          A�R@��@�  >�z�?�  B��@��@�����7�B�R                                    Bx���  �          A;�
A33@�z�����4��B\)A33@�  �Mp���33B                                      Bx���  9          A8z�A@ڏ\�`����Q�B
=A@�Q���ff��z�Bff                                    Bx��<  =          A1G�@�33@�����ff�G�B!\)@�33@��
��  ��B=q                                    Bx���  T          A6�\A�H@ȣ���(��ʣ�B  A�H@�Q���
=���B��                                    Bx�Ĉ  �          A5�A{@���`  ���Bz�A{@Ǯ��{��(�B                                    Bx��.  �          A7\)A��@�(��'��T  B33A��@��g�����B�H                                    Bx���  T          A4��A	�@�����)p�B�A	�@�{�E��|��B�
                                    Bx��z  T          A.�RA
=@���G��*�HB ��A
=@�G��@  ��B�                                    Bx��   �          A*�\@��@��H���J�RB,�R@��@��U���\B%�
                                    Bx��  �          A-p�@�  @��
�����
B.G�@�  @����6ff�s�B(�
                                    Bx�l  
�          A1�A   @�
=��ff��B,�A   @��
�8Q��pQ�B&�R                                    Bx�+  T          A=p�A�
A�\������{B6=qA�
A�\�33�3\)B2��                                    Bx�9�  �          AH��A��A�Ϳ�G���(�B3ffA��A��2�\�M��B/�                                    Bx�H^  �          AV�\AQ�Aff�@  �O
=BE
=AQ�Ap���{���HB>p�                                    Bx�W  �          AVffA�A!���E�UBK=qA�AQ���=q���BD��                                    Bx�e�  T          AT��A   A"=q�E��V�\BN�A   A����=q����BH\)                                    Bx�tP  �          @�ff@��H�����"�\��=qC��@��H��(���؏\C���                                    Bx���  �          @�z�@z��c33���H����C�"�@z��qG���z��p��C�ff                                    Bx���  �          A�R@�ff��ff�Dz���G�C���@�ff���H��
�P(�C���                                    Bx��B  k          A\)?�\)�Q���H�S�
C���?�\)�  ?G�@��C���                                    Bx���            A�\��=q�ff���J=qC�1쾊=q���?���A z�C�/\                                    Bx���  �          A�׾�\)�
=@�AX��C����\)� z�@S�
A�33C�                                      Bx��4  �          AG������?��A��C��þ��(�@(��A��C���                                    Bx���  �          A�p���Q����QG�C�.�p����
?p��@�(�C�+�                                    Bx��  
�          A\)��
�  ��z��"ffC�{��
�ff�u����C�\                                    Bx��&            Ap��(���
��z��>=qC~h��(��
=���Mp�C~�R                                    Bx��  o          A(�����ÿ�{� ��CxR���33�W
=���
C�3                                    Bx�r  T          A
=�{�  �%��{C}O\�{��Ϳ�G�� ��C}޸                                    Bx�$  "          Ap��7����%�s�
C|s3�7��녿�\)����C|�R                                    Bx�2�  T          Az��8���G��
=�G�C|Y��8����ÿ#�
�o\)C|�q                                    Bx�Ad  T          A�R�<���  �����:ffC{�H�<���33���H�9��C|=q                                    Bx�P
  
�          A�R�)����\��p����C}�
�)���Q�<#�
=�\)C~�                                    Bx�^�  �          Aff�  �z῞�R���
C�1��  �>�\)?�33C�@                                     Bx�mV  �          A���G
=��׾�(��)��Cz}q�G
=�  ?�  @��
Czff                                    Bx�{�  �          A=q�S33�Q���Ϳ!G�Cx���S33��R?��
AG�Cx��                                    Bx���  �          A  �N�R�=q>�p�@ffCx��N�R��?�p�A1��Cx��                                    Bx��H  
�          A���Dz���
?
=@o\)Cz��Dz��z�?�(�AIp�Cy�f                                    Bx���  �          A��_\)� (�?�Q�A.�RCvs3�_\)���
@>�RA�z�Cuc�                                    Bx���  
�          AG��`  �p�@�AX(�Cv���`  ��(�@Z�HA��CuQ�                                    Bx��:  �          A��Tz����?���@��CyJ=�Tz���@(��A��Cx��                                    Bx���  �          A�R�!G���?���@��C~T{�!G��
�R@'
=A��HC}��                                    Bx��  �          A\)��ff���=���?
=C�b���ff�=q?�Q�A!G�C�P�                                    Bx��,  �          A#��
=�\)����+
=C�O\�
=�"{�B�\���C�W
                                    Bx���  
�          A33��z���R�����6=qC�#׾�z����z���HC�(�                                    Bx�x  �          A�þ��ff��Q��
=C��3����ͽ#�
�aG�C��R                                    Bx�  T          Ap�@2�\���������>  C���@2�\�������H��C�n                                    Bx�+�  "          A��@ ����\)�׮�Wz�C�XR@ ����=q��p��4��C���                                    Bx�:j  �          A�׿ٙ�����[���{C���ٙ��	����R�M�C�aH                                    Bx�I  �          A33����R������=qC�3�����H�@  ��\)C�:�                                    Bx�W�  �          A�
@.�R�z����L�C��
@.�R�g���\)�g�\C���                                    Bx�f\  
�          AQ�>Ǯ�ƸR��Q��0  C��>Ǯ�����33�=qC���                                    Bx�u  
�          A�H���H��z���  �#=qC��H���H��G���Q�����C�"�                                    Bx���  
�          A
=>k����H���H�&��C��R>k���G���������C���                                    Bx��N  �          Az�#�
��Q���  �Q�C�ٚ�#�
�	�����  C��q                                    Bx���  �          A������R�����\C��f���G��0�����HC�&f                                    Bx���  �          Ap����
�ff���
���Cs0����
��׽�\)��Cs�
                                    Bx��@  �          A��;����Q��p��Cz���;��
�\�^�R���C{O\                                    Bx���  
�          A�?�z����
����=qC���?�z���(��c�
��  C�
                                    Bx�ی  
�          A_\)A �׿�ff�4  �W�C��\A ���q��,  �I�RC�h�                                    Bx��2  "          Ad(�A �Ϳ���8���Y��C���A ���u��0���LG�C�J=                                    Bx���  �          Ad��@�zῦff�=�`  C�W
@�z��Y���6�H�T=qC�U�                                    Bx�~  T          AdQ�@�Q�����=G��`�C�e@�Q��{��4���QC���                                    Bx�$  "          AdQ�@�{�  �<Q��_  C���@�{���H�2�R�N��C�G�                                    Bx�$�  T          Ad��@����{�>�H�b�\C�  @���`���7�
�V{C���                                    Bx�3p  �          AdQ�@��
��G��?��d�RC�Q�@��
�Z�H�8���Xp�C��                                    Bx�B  9          Ae�A�Ϳ����8���WQ�C��fA���j=q�1��J�C�q                                    Bx�P�  o          Ae�A33�����8���U�
C�s3A33�\���1���JG�C���                                    Bx�_b  T          Aep�A�׿��7
=�S�\C�FfA���`���/��G��C���                                    Bx�n  
q          Ad(�A(������5p��R�C�,�A(��aG��.{�G  C�                                    Bx�|�  o          Af{A�׿�ff�5G��O��C��{A���H���/
=�F
=C�+�                                    Bx��T  "          Ab�HA�
��
=�-��H(�C�|)A�
�[��&�\�<��C��\                                    Bx���  "          Ac�A����p��,���E�\C���A���n�R�$���933C��q                                    Bx���  "          Ac�A����z��/��J�\C���A���l(��'��>
=C��\                                    Bx��F  �          Ac�
Az����)��@{C���Az�������
�2�C�AH                                    Bx���  �          Ab�RA
=���H�+33�D\)C�AHA
=�Mp��$z��:33C�`                                     Bx�Ԓ  �          Aa��A  ��Q��6�\�XffC��qA  �!G��2=q�P�C�~�                                    Bx��8  �          A`��@��
���9���_G�C�>�@��
�+��4���V��C��{                                    Bx���  
�          A_�@�G�?J=q�F�\�{@�ff@�G��˅�EG��y  C���                                    Bx� �  T          A[33@�(�>W
=�A���x�R?��@�(����>�R�rG�C��{                                    Bx�*  T          A]��@�\��\)�>=q�l��C��3@�\�
=�:ff�ez�C���                                    Bx��  �          A^ff@��
���
�733�]�C���@��
�!��2�R�V
=C�R                                    Bx�,v  �          A[�
A33�#�
�*�R�K=qC��{A33���'��E�C�C�                                    Bx�;  T          AW�
@��i���6�\�g\)C���@������'��L33C�h�                                    Bx�I�  T          AC�@ȣ���$z��d��C���@ȣ������R�Q=qC���                                    Bx�Xh  �          AF�R@�{��
�&{�bz�C�%@�{�������M�C�#�                                    Bx�g  �          A<(�@�z��b�\�33�^(�C��{@�z���(�����B{C�/\                                    Bx�u�  T          A;�@�=q���
��U�HC��@�=q��(���7�\C���                                    Bx��Z  T          A;33@�(��4z����[=qC���@�(�����G��C�\C�b�                                    Bx��   T          A:�R@�G��'
=�{�^�\C�N@�G���\)��R�Gp�C��)                                    Bx���  
�          A7�@���H���z��X��C��@�������>�C��                                    Bx��L  �          A(z�@�Q����H�	p��Z�\C���@�Q���
=��  �5z�C�e                                    Bx���  T          A�
?��dz���\)�bp�C��f?���33�����4�C��f                                    Bx�͘  �          A
=@
�H��(����\�)��C�t{@
�H�����|�����\C��)                                    Bx��>  
�          A�H@����H��z��#��C�p�@����H���R�{C�K�                                    Bx���  �          A�\@,����
=����V=qC�Ff@,����p��Ϯ�+��C�k�                                    Bx���  �          A!��@Mp����
����E�C�q�@Mp��׮�������C���                                    Bx�0  �          A�@�
��
=���
�]�C��@�
����  �2G�C�>�                                    Bx��  �          A�
@�H������^�C��@�H��z�����3=qC�H                                    Bx�%|  k          A)�@ �������
=�\=qC�5�@ ����ff���
�0{C��)                                    Bx�4"  �          A&=q@>{���H�{�W��C��@>{�޸R��33�,�RC��3                                    Bx�B�  "          A#\)?���ᙚ�ᙚ�0�C�|)?�������{��C�Z�                                    Bx�Qn  "          A#�?޸R��33�陚�7�RC�"�?޸R����
=���C��)                                    Bx�`  �          A=q?����
��p��+  C���?����G�����C���                                    Bx�n�  T          A%p�?�=q�Ϯ����?��C��{?�=q������z��C��                                    Bx�}`  �          A*�H@��
��ff�Q��S�C���@��
������ff�1{C���                                    Bx��  �          A*�\@��R���	��V��C���@��R��(������0G�C��3                                    Bx���  �          A$��@QG���
=�G��N�C�n@QG�����׮�#�C�o\                                    Bx��R  �          A)�@:�H����
ff�[33C�K�@:�H��=q��=q�.�C�8R                                    Bx���  �          A*=q@'���(��
=�o��C���@'���\)����CQ�C�                                    Bx�ƞ  �          A'�@G��aG�����}�C��=@G����{�T�
C���                                    Bx��D  �          A$��@e�w��
ff�j�\C�j=@e��(���(��C33C�>�                                    Bx���  "          A%�@y���33��R�~��C��
@y����
=�\)�_ffC�]q                                    Bx��  �          A(��@�p��*�H����l��C�q@�p�����(��N��C���                                    Bx�6  T          A+\)@����^�R�z��f\)C�L�@�����33� ���C�RC�33                                    Bx��  �          A)��@��H�X������j��C���@��H�����p��G\)C���                                    Bx��  �          A(��@{��J=q��
�u��C���@{����H�p��QffC��3                                    Bx�-(  �          A/\)@�z��|�����e�C��@�z����
�=q�@
=C�j=                                    Bx�;�  "          A1G�@���vff�Q��d=qC�{@�������33�?��C�G�                                    Bx�Jt  
�          A/�
@��R�����o�
C�o\@��R��33���S�C�^�                                    Bx�Y  T          A7�@���O\)�z��jC��H@����=q�G��IC��R                                    Bx�g�  �          A8(�@�\)�k�����jQ�C�\@�\)��Q��  �F\)C��{                                    Bx�vf  T          A7�@�p��R�\�G��k�HC��@�p���z���J�C�/\                                    Bx��  �          A6{@����C�
����m�HC�@���������M
=C���                                    Bx���  o          A8Q�@�33�?\)�  �gz�C�e@�33���H�p��H�C�4{                                    Bx��X  �          A6�H@�=q�u���`�C�n@�=q������<�HC�l�                                    Bx���  "          A733@�\)�g
=��R�f{C��@�\)��{�	�B��C���                                    Bx���  �          A2ff@�=q�xQ��{�e�C���@�=q��z��(��?C��                                    Bx��J  �          A.�R@�p������  �gQ�C���@�p���G��p��?33C�Q�                                    Bx���  �          A+�
@~�R��\)�z��e  C���@~�R������H�;�C�p�                                    Bx��  T          A*{@xQ���\)�Q��_�
C�u�@xQ��˅����533C���                                    Bx��<  �          A&=q@��H�'����g{C�@��H���������G�
C�c�                                    Bx�	�  �          A)��@�  >�G��
�H�]=q@��@�  �Ǯ����XQ�C��3                                    Bx�	�  �          A'\)@�p��s33���bz�C�B�@�p��:�H�Q��Qz�C�\)                                    Bx�	&.  �          A"{@��
��ff��R�Y  C�˅@��
�z���33�L�
C�4{                                    Bx�	4�  �          A z�@���޸R����_33C���@���g
=����Gz�C�                                    Bx�	Cz  �          A"=q@�{�aG�����P�HC�  @�{��
��
=�G=qC�˅                                    Bx�	R   �          A"�H@��H>�ff��\)�E�H@z=q@��H��{���
�B=qC�/\                                    Bx�	`�  "          A"=q@�z�?\(���33�B�\@�33@�z�p�����H�B=qC���                                    Bx�	ol  �          Aff@�{�������offC���@�{�%���H�_
=C�1�                                    Bx�	~  �          A"�H@�Q�>���\)�W�@ ��@�Q��� ���P�
C�>�                                    Bx�	��  �          A%G�@��H>��{�O�@�@��H���H� (��K�C���                                    Bx�	�^  �          A#\)@�33��G��p��[�RC�xR@�33�z��G��Q�
C�AH                                    Bx�	�  
�          A!�@��
�\)�G��_\)C�+�@��
� �����R�Q{C��R                                    Bx�	��  T          A!G�@����\��`{C�` @����R����R
=C��                                    Bx�	�P  
�          A!@�=q�+����Z(�C��{@�=q�'
=���H�K\)C���                                    Bx�	��  T          A!@��H��G��G��^  C���@��H�Mp���G��I�C�{                                    Bx�	�  �          A ��@�33�ٙ��33�f=qC��=@�33�j�H�����L��C��                                    Bx�	�B  �          A (�@�\)��p�����`{C���@�\)�z�H��=q�D�C��3                                    Bx�
�  �          A$Q�@��
�
�H�ff�p��C�l�@��
��  ���QG�C���                                    Bx�
�  
�          A$z�@�=q�&ff�  �w
=C��R@�=q��{����R
=C�y�                                    Bx�
4  
�          A%p�@�  ������t�
C�g�@�  ��  �z��T��C�W
                                    Bx�
-�  �          A'�@�=q�  �Q��m�HC���@�=q���
�33�NQ�C��                                    Bx�
<�  �          A'\)@��R�Q����q(�C��3@��R�����z��QC��                                    Bx�
K&  
�          A'�
@����C�
����o{C�E@������ ���H33C��R                                    Bx�
Y�  T          A(z�@�(��Mp��Q��k��C��@�(���=q���R�Dz�C��q                                    Bx�
hr  "          A(  @�=q�W
=�G��oQ�C�9�@�=q�����\)�E�RC�H�                                    Bx�
w  �          A$Q�@Z=q�������iffC���@Z=q��p����9��C�q�                                    Bx�
��  o          A#\)@E��  ���a
=C���@E��{�ᙚ�.��C�g�                                    Bx�
�d  �          A"�H@>�R��p���R�^�C��
@>�R��33��{�+��C���                                    Bx�
�
  T          A�R@��\�Z=q�(��e  C�\@��\��33�����;{C���                                    Bx�
��  �          A!p�@���7
=���cQ�C�8R@����33��z��>�RC��=                                    Bx�
�V  �          A"�\@C33���R�   �O�HC��R@C33������(��C�]q                                    Bx�
��  �          A!��@5���� z��R��C�ٚ@5��  ��p���C��{                                    Bx�
ݢ  �          A"�\@P�����R��
�X�C��@P�����
��\)�&{C���                                    Bx�
�H  T          A%p�@h����\)����`
=C���@h���Ϯ����/�\C��                                    Bx�
��  �          A$Q�@e���=q�\)�^(�C��@e���������-{C�S3                                    Bx�	�  �          A$��@�  �����	p��b33C�t{@�  �����G��4p�C���                                    Bx�:  �          A%�@��H�}p��
�\�b�C��{@��H�������
�5Q�C��                                    Bx�&�  �          A(Q�@~�R��z��
�R�^��C�
@~�R��ff����/33C��{                                    Bx�5�  �          A$Q�@��R�n�R�	G��b��C�7
@��R�������H�6��C��q                                    Bx�D,  T          A"�\@����z�H���a�HC�޸@�����ff���4
=C��                                    Bx�R�  �          A"�H@����fff�  �b��C���@��������G��7(�C��                                    Bx�ax  �          A"ff@���S�
���b�C�Ǯ@����(����H�9=qC��=                                    Bx�p  �          A=q@l(��|���  �c  C��=@l(���{��ff�3=qC��                                    Bx�~�  �          Aff@>{��z�� z��bQ�C�
=@>{��=q���
�.  C��
                                    Bx��j  
(          A{@�R��(�����XQ�C�t{@�R��p������!�C�/\                                    Bx��  T          AQ�@Q��{����d�C��@Q������z��2�RC��)                                    Bx���  "          Ap�@dz���(�����X��C�h�@dz����R��
=�'ffC�u�                                    Bx��\  
�          A�@5���G������L(�C���@5��׮���H���C�j=                                    Bx��  
�          AQ�@1G���{��  �W�HC�E@1G��Ϯ��  �!�C���                                    Bx�֨  "          A(�@Y���i�����R�dp�C�� @Y����G���\)�3��C���                                    Bx��N  
�          A��@XQ���
=����]G�C��@XQ���{��z��)z�C��R                                    Bx���  �          A(�@Vff��(�� ���^\)C��
@Vff��33�Ӆ�*�C��=                                    Bx��  
�          Az�@4z����R���R�a�
C��@4z������  �+ffC���                                    Bx�@  �          A�@]p��������H�W�C���@]p���ff�˅�#�C�q                                    Bx��  T          A�
@A���(�����WffC���@A���G������ �C��                                    Bx�.�  �          A��@1��������L�C��\@1����H��
=��C���                                    Bx�=2  T          A��@,(�����陚�N��C��@,(����
����
=C���                                    Bx�K�  T          A�@�R��\)���Lz�C���@�R�޸R������C��                                    Bx�Z~  �          A=q@0�����
�ۅ�B�C��
@0����Q���p��	C���                                    Bx�i$  S          Ap�@9�����H��33�E�HC�@ @9���ᙚ�����G�C�*=                                    Bx�w�  �          Ap�@>{��  ��(��G�C��)@>{��\)��{�C���                                    Bx��p  "          A33@C33�����R�G��C��@C33�߮�������C�Ǯ                                    Bx��  "          A�H@�������Kz�C��@���R�����C�N                                    Bx���  �          AQ�@/\)��Q���ff�9��C��R@/\)���
��(����RC�33                                    Bx��b  �          AQ�@;����\�љ��4ffC�T{@;��������R��
=C���                                    Bx��  �          A@0����\)���
�<�
C��
@0��������G��Q�C�9�                                    Bx�Ϯ  "          A�@8�������z��D�C�s3@8������{�
�C�T{                                    Bx��T  "          A�\@L(���z�����A(�C�L�@L(��������\)C��                                    Bx���  �          A
=@:=q��
=�����>G�C��q@:=q��(���Q��=qC�R                                    Bx���  �          A{@0  ��\)��
=�F�
C��H@0  ��ff��  �p�C���                                    Bx�
F  
Z          A�
@<�����H���
�I�RC�\@<���ۅ���(�C���                                    Bx��  
�          A��@A��������I=qC�� @A���(���33�\)C���                                    Bx�'�  �          A�@:=q��\)��p��Dz�C�q�@:=q��=q��G��	Q�C��H                                    Bx�68  
@          A z�@9����p����Cz�C��@9����G���33��HC�7
                                    Bx�D�  <          A�@L(�������G��G�C�f@L(������ff���C���                                    Bx�S�  �          Az�@Fff��ff����AC�B�@Fff�������
C�0�                                    Bx�b*  �          A{@P  ��33��
=�D\)C��@P  ��\)��33�	��C��q                                    Bx�p�  �          A��@\�����R��{�<�C��\@\����  ������RC�W
                                    Bx�v  �          A�@\����\)����?�RC��f@\�����H�����\C�9�                                    Bx��  T          A�\@hQ���\)��p��A�RC��H@hQ���33������C�"�                                    Bx���  �          A�\@a���{��
=�V�C�xR@a���Q������\C���                                    Bx��h  �          Aff@c�
��ff�����Oz�C��H@c�
��ff������C���                                    Bx��  T          A!�@a���z����Bz�C�@a�����(��(�C��                                    Bx�ȴ  �          A�@\(����R����=�
C��@\(���G����H�
=C�E                                    Bx��Z  �          A��@\(����H���\�W{C�ff@\(���z���{�z�C��\                                    Bx��   �          A(�@o\)��  � ���_G�C���@o\)��z��ҏ\�)�
C���                                    Bx���  �          A��@��\��G������R�
C�|)@��\�Å��G�� (�C���                                    Bx�L  �          A@�Q���G������PQ�C��@�Q��Å��G��p�C�<)                                    Bx��  �          A��@���p�����SC��@���  ������C��H                                    Bx� �  �          A(�@�(����R��(��MC�j=@�(���
=��Q��=qC�G�                                    Bx�/>  �          Az�@�G����\���
�M
=C���@�G����H���R�z�C���                                    Bx�=�  �          A (�@R�\������=q�M��C��R@R�\��\��{�  C��                                    Bx�L�  T          A (�@C�
��Q������K�HC��f@C�
�������\���C�                                    Bx�[0  T          A�@U��\)���](�C�U�@U��z�����#{C�XR                                    Bx�i�  
�          A�R@j�H�w��p��e�C�@j�H��(���33�/{C�q�                                    Bx�x|  �          A33@)���|�����u��C��\@)���ʏ\���9=qC�\)                                    Bx��"  �          Az�@Dz��tz����o��C�ff@Dz���z���
=�5�HC�J=                                    Bx���  T          Az�@Y���@  ���v(�C�K�@Y�����\���
�Az�C�Ff                                    Bx��n  	`          A33@�녽������iQ�C�H�@���=q��p��X(�C���                                    Bx��  �          A  @�p�?�����l��A�p�@�p��=p�� ���s�\C�:�                                    Bx���  �          A��@�  @���z��U�HA��@�  �u���a��C���                                    Bx��`  T          A\)@�
=@��H��  ��
B1  @�
=@E����H=qA�{                                    Bx��  �          AG�@��R@��R���
�z�BD{@��R@w
=�ڏ\�=z�Bp�                                    Bx���  �          A�@�{@��R���\�
=B9�H@�{@c33��ff�?z�B	�\                                    Bx��R  	�          A#�
@��H@�{��  �%��B �@��H@,������K�AƏ\                                    Bx�
�  T          A%�@��@�
=��\)�1=qB��@��@
=��
�R�A���                                    Bx��  "          Aff@��@�G���
=�#�HB=q@��@����G��G�A�ff                                    Bx�(D  �          A$z�@�  @�\)������HB*  @�  @n�R��\)�3{A���                                    Bx�6�  �          A!�@�(�@������(�B.�R@�(�@}p���p��-��BG�                                    Bx�E�  �          A$z�@��
@����p��{B�@��
@E����2  A�                                    Bx�T6  �          A(z�@�=q@�G������  B<p�@�=q@��H��z��,��Bp�                                    Bx�b�  �          A�@�p�@ҏ\��p����
BJ33@�p�@��R��p��/�
B#��                                    Bx�q�  
�          A=q@���@�  ��Q��ffBQ33@���@�����
=�;��B'�                                    Bx��(  �          A{@��R@��\��=q����B;��@��R@��H���
�,(�Bff                                    Bx���  T          A33@���@�=q���H���B9�
@���@e��\)�4�B�\                                    Bx��t  �          A�R@��
@�Q������HB1=q@��
@3�
��Q��@  A�z�                                    Bx��  �          A�R@���@�=q���
����B/�H@���@i�������)Q�B
=                                    Bx���  "          A@�  @�  �H�����
B/��@�  @���������B
=                                    Bx��f  T          @��@���@�녿Tz�����B4
=@���@���ff��=qB(�                                    Bx��  �          @Ӆ@��
@XQ��7
=���
B
�@��
@�p���G�A̸R                                    Bx��  
�          @�\)@���@���8����33A�ff@���?����_\)�
=A���                                    Bx��X  �          @��@W�?�Q��'
=���A���@W�?\)�;��"Q�A�                                    Bx��  �          @��@;�@��G�����Bp�@;�?�
=����ffA��                                    Bx��  �          @�z�@��H@��\�{��G�B?(�@��H@�����  ����B&��                                    Bx�!J  T          A\)@�p�@�  �\����33BD\)@�p�@��
��
=���B'                                    Bx�/�  �          A�H@���@��\�����RBN�
@���@��������HB3��                                    Bx�>�  �          A{@�\)@�(��mp���33B@(�@�\)@���������RB"{                                    Bx�M<  �          A�
@�(�@��
���H��=qBB�@�(�@�ff�����#G�Bz�                                    Bx�[�  �          A��@�p�@�p�������\B?=q@�p�@_\)��\)�@�B�                                    Bx�j�  �          A@��R@�����
�#�B)33@��R@%��陚�M��A���                                    Bx�y.  
�          A@�@�����  ��HB0��@�@R�\��(��=33A�\)                                    Bx���  �          A�\@���@������Q�B,{@���@HQ���ff�>Q�A�{                                    Bx��z  �          Ap�@���@�����G��{B+  @���@9������A��A�                                    Bx��   �          A{@�(�@�Q�����p�B+z�@�(�@1���
=�D
=A�                                    Bx���  �          A�@���@�{��  �G�B2{@���@L(��ۅ�@�A���                                    Bx��l  T          A�@��R@��
�����(�B,�@��R@H���׮�;��A�                                    Bx��  �          A=q@�=q@�(���(��\)B*z�@�=q@N�R�Ϯ�4�
A��H                                    Bx�߸  "          A
{@��@����ff�陚B>��@��@{���  �'�Bp�                                    Bx��^  �          A��@��\@��\��
=���B!��@��\@!G����H�<�Aң�                                    Bx��  "          A�R@���@�\)�������B&�@���@)����{�=�RA�{                                    Bx��  "          A��@���@�����
=�  B3ff@���@P  �\�5\)B
=                                    Bx�P  �          A��@���@�G���=q��B1�@���@S�
��ff�0��B��                                    Bx�(�  �          A�@�Q�@��������
B@  @�Q�@tz���G��/33Bff                                    Bx�7�  "          A(�@��@>�R��
=�2(�A��R@��?z�H���
�M�A1                                    Bx�FB  �          A�@�\)?�z���=q�O{A|(�@�\)�J=q��p��S�C��H                                    Bx�T�  "          A(�@��@,���љ��JQ�A���@��>�G���=q�b
=@�z�                                    Bx�c�  �          A	��@�{@Y����p��?B@�{?�����`Q�AZ�\                                    Bx�r4  �          A  @�(�@z��Ϯ�H�A�p�@�(�=�Q���z��Z��?��                                    Bx���  �          A	��@�  @>{��ff�L��B�@�  ?#�
�陚�hQ�A z�                                    Bx���  T          A  @���@)�����
�N{A�{@���>�33��(��e�@�ff                                    Bx��&  T          A\)@��?�������j33A���@����ff��33�l��C��
                                    Bx���  "          Ap�@o\)���
��33�n�
C�G�@o\)��  ��(��@��C���                                    Bx��r  
�          A=q@�(�>�
=��z��\{@��@�(����
����Q33C��                                     Bx��  �          AQ�@���#�
����n
=C��{@���ff����Y\)C�>�                                    Bx�ؾ  
(          A�@\�Ϳ���  � C�Ff@\���:�H����_\)C��                                     Bx��d  "          A	��@mp����
��  �)C�� @mp��#�
����ip�C��                                    Bx��
  �          A
{@I��>���� (��R@�Q�@I�������{�}
=C�33                                    Bx��  �          AQ�@Z�H?���\)�qA	@Z�H��
��
=�wp�C�|)                                    Bx�V  �          A�@]p��#�
�����C��@]p���H���
�i��C��=                                    Bx�!�  
(          Ap�@X�ÿ+��陚ffC�n@X���B�\��p��^�RC�f                                    Bx�0�  �          A�@XQ�#�
���H��C��=@XQ��A��ָR�_�RC��                                    Bx�?H  T          A ��@Y���!G���\)�fC���@Y���@  �Ӆ�^Q�C�L�                                    Bx�M�  �          A��@Z�H�Q���Q�8RC�B�@Z�H�L(���=q�Z=qC���                                    Bx�\�  T          @�p�@7�������(��RC��@7��w
=��ff�P�HC�S3                                    Bx�k:  �          @��H@8���9�����
�h�C�t{@8����{�����*�C�+�                                    Bx�y�  �          @�  @>{��G��޸R�=C���@>{�Q���\)�Z�C�
                                    Bx���  �          @�\)@�R>#�
��=q
=@�
=@�R�Q���{33C���                                    Bx��,  �          @�p�@���
=��ff\C�@��333�����tQ�C�R                                    Bx���  n          A�H@#33��\)��\u�C�Ф@#33�.�R����u�C��                                    Bx��x  "          Aff@C33�C33��p��h=qC�xR@C33��ff����)Q�C�.                                    Bx��  "          A ��@[���\���H�kC�)@[����R��p��4��C�                                    Bx���  
�          Az�@y������p��o�\C��)@y���p���ȣ��D��C�                                    Bx��j  �          @�ff@tz�������c=qC��
@tz����R����1�C�!H                                    Bx��  
�          @��@�z��	���ȣ��PQ�C���@�z���z����"z�C�T{                                    Bx���  
�          @��@Vff�6ff��=q�X\)C�˅@Vff�����  �(�C��                                     Bx�\  n          @��@,(��Q��\�\Q�C���@,(�������33���C��\                                    Bx�  
�          @��@W��<(�����PQ�C�q�@W����R������
C��f                                    Bx�)�  
�          @�Q�?^�R������
=�F��C�&f?^�R���p  ��ffC��
                                    Bx�8N  T          @�p�?=p���G���{�@C�4{?=p���p��hQ���p�C�/\                                    Bx�F�  �          @�G�?�����  ��p��`p�C���?���������z�C�/\                                    Bx�U�  �          @��R?������θR�c�C�Ǯ?�����
�����z�C��{                                    Bx�d@  �          @��H>8Q��tz��ڏ\�r��C�W
>8Q�������"��C��q                                    Bx�r�  �          @�Q�=�\)�s33�أ��r�HC�� =�\)��(�����"�RC�Q�                                    Bx���  T          @�>��H����Q�aHC��>��H��(���G��Q\)C�                                    Bx��2  
�          @�\)?����^{�Ӆ�t�C��
?���������Q��&�\C�~�                                    Bx���  T          A�?n{��\)��p��T33C��?n{������z��C���                                    Bx��~  T          Ap�?����33�ȣ��B��C�޸?����������  C�(�                                    Bx��$  �          A{?Y��������ff�,
=C���?Y�������P����C�'�                                    Bx���  �          A�R?\(������{�H�\C���?\(���R���R���HC�ff                                    Bx��p  T          A=q?E����H��=q�F�C�q?E���{���\��p�C�\                                    Bx��  �          Ap�?@  ��33������C�` ?@  �����0  ����C��H                                    Bx���  "          A��?��\�ҏ\���\�(�C�z�?��\��33�  �|z�C��)                                    Bx�b  :          A
=?����ff�hQ���=qC�1�?��� (��fff��z�C��f                                    Bx�  "          A��?�p����H�5��33C���?�p�����
��C�,�                                    Bx�"�  �          A�
@
=���������((�C��)@
=��  �O\)���RC��                                     Bx�1T  �          A  ?���ff��p��)33C��{?�����O\)����C���                                    Bx�?�  T          A=q?xQ���p��������C��
?xQ��33��(��!G�C�]q                                    Bx�N�  �          A��?�(��������H�+�C�L�?�(����N{��z�C��3                                    Bx�]F  �          A��?�ff���H���\�\)C��?�ff��
=�!����C���                                    Bx�k�  �          A
=?����������\�#�C���?�����z��:�H���C��                                    Bx�z�  "          A�H@����H��
=�1�\C��
@���  �^�R�ɅC�W
                                    Bx��8  "          AQ�?��H��{��(��,G�C�~�?��H����QG����RC��                                     Bx���  �          A��@1G����\���R�.Q�C��R@1G���  �^{��  C���                                    Bx���  �          A��?�\)��
=���%
=C���?�\)��  �>{��p�C��                                    Bx��*  T          A��?�(�������33�)��C�h�?�(���z��L(���p�C�xR                                    Bx���  �          A  ?�
=��\)���R�z�C�P�?�
=��
=���n{C��                                    Bx��v  T          A�
@p���z���  �Q�C�K�@p����
�4z���33C�\)                                    Bx��  
�          A��@+��������*�C�/\@+������R�\����C�p�                                    Bx���  �          A(�@��θR��ff��C��@���G����(�C�W
                                    Bx��h  "          A�?�ff���H�S33����C���?�ff�Q�Ǯ�)��C�w
                                    Bx�  :          A(�?s33��z���33�"��C�c�?s33��z��4z����RC��=                                    Bx��  �          Ap�?�G��ڏ\���ffC�>�?�G���
=���H�>ffC�aH                                    Bx�*Z  �          A��?ٙ��ٙ���ff��\C�f?ٙ���{��p��@Q�C��                                    Bx�9   �          A�\?�ff��{���
��C��H?�ff��녿ٙ��@(�C��R                                    Bx�G�  �          A ��?����p�������C���?������Q���  C�Ǯ                                    Bx�VL  
�          A=q?����=q��\)����C�t{?����(���G��+
=C��)                                    Bx�d�  �          A�R?.{��Q��j�H��{C���?.{�p��Tz����HC�e                                    Bx�s�  
�          A�\?�=q���H�w
=��z�C�S3?�=q� (������C���                                    Bx��>  �          A��?�������Q����HC���?��� zῙ���ffC��=                                    Bx���  T          A{@C�
��z�������\C�C�@C�
��ff����z�\C�/\                                    Bx���  
�          A=q@g���ff��ff�{C�1�@g���33�$z�����C�~�                                    Bx��0  T          A�@�Q������(���RC�AH@�Q�����&ff��C�=q                                    Bx���  �          A ��@�����\)�33C�b�@��ff�(��}�C��
                                    Bx��|  �          A z�@ ����\)��
=��C�L�@ �����
�{���C���                                    Bx��"  �          A{@+�������
=�!Q�C��@+��ᙚ�7���z�C�l�                                    Bx���  �          A�@0����\)��{�G�C��q@0����(��!���ffC���                                    Bx��n  �          @��@(��������\�33C���@(������.{��p�C�K�                                    Bx�  �          @���@6ff��\)��p��#�HC�K�@6ff�׮�:�H���C�q�                                    Bx��  �          @��
@L(���G����
�,33C��3@L(�����QG���{C�5�                                    Bx�#`  �          @��\@[����\��=q�+�C�l�@[���ff�R�\��G�C�z�                                    Bx�2  �          @��@e��(�����G�C�,�@e�ʏ\�333��z�C��f                                    Bx�@�  �          @�33@j=q����z��#��C��@j=q��\)�E�����C�4{                                    Bx�OR  �          @�@[�����(����C��H@[��Ӆ�)������C���                                    Bx�]�  �          @��R@Z=q��G���\)��C�ff@Z=q�����{��  C��3                                    Bx�l�  �          @��\@����y�����R�0��C���@������
�i������C�˅                                    Bx�{D  �          @�(�@��H�G������=z�C�"�@��H������Q���RC�c�                                    Bx���  �          @�@�(��hQ���{�8  C�^�@�(����R�~{��RC���                                    Bx���  �          @�@���w������1�C�XR@����(��n{��RC�\                                    Bx��6  �          @�@u�������
�4�RC��@u����H�n�R����C���                                    Bx���  T          @�p�@����w������0C��@������
�mp��ߙ�C�9�                                    Bx�Ă  
�          @���@�{�W
=����;��C��f@�{��\)��33���C�W
                                    Bx��(  �          @�z�@�(��Mp���
=�:  C��@�(����H��z���33C�^�                                    Bx���  
          @�33@u�e������?\)C�� @u��
=��=q����C��=                                    Bx��t  T          @�Q�@c33���\���
�/{C���@c33��Q��X����G�C�K�                                    Bx��  �          @�{@U��  ��ff�z�C�:�@U�У�������
C���                                    Bx��  
�          @�{@ ��������\�	��C���@ ����  ����b�RC���                                    Bx�f  �          @�\)@���
=��{�ffC��\@����H��
=�J�\C�8R                                    Bx�+  T          @�
=@p������Q��ffC�^�@p���׿���XQ�C���                                    Bx�9�  
�          @�
=?�33���
����
C�S3?�33�ᙚ�\)����C���                                    Bx�HX  �          @�@33��Q���=q��\C�U�@33���������C�7
                                    Bx�V�  T          @�z�?�p������{�!ffC���?�p��߮�!����\C��                                    Bx�e�  "          @�?���{���\�'
=C��?����,(���p�C���                                    Bx�tJ  �          @�p�@)�����
����p�C���@)����(����{�C��                                    Bx���  �          @�@"�\��Q���z���C��@"�\��\)��
=�iG�C���                                    Bx���  �          @�z�@!G���������  C��R@!G���\)��ff�YC��=                                    Bx��<  
�          @��@5���������C�@ @5����
�����_�C�5�                                    Bx���  T          @�p�@\)��z���33�{C�e@\)����\)����C��                                    Bx���  �          @�
=@���
��Q��633C��R@��=q�R�\��C��                                    Bx��.  
�          @���@���\)��p��@=qC�:�@������e��Q�C��3                                    Bx���  �          @��@%��������2  C�3@%���
=�J�H���
C�ٚ                                    Bx��z  T          @�R@z������(���\C��
@z��ָR������C��                                    Bx��   
Z          @��@����������"�C��@��θR�"�\���
C�S3                                    Bx��  
�          @���@\)��{�����=z�C���@\)��{�]p���33C��R                                    Bx�l  "          @�
=@���p�����/
=C��3@���  �;����C�Ff                                    Bx�$  "          @�
=@�����33�"Q�C�� @���(��#33���C�~�                                    Bx�2�  �          @�Q�?��R��33���H���C�!H?��R��녿����b=qC��{                                    Bx�A^  T          @�
=?h�������n�R��{C��?h�����H��ff��C���                                    Bx�P  T          @���    ��33�=p�����C���    ��R�#�
����C��)                                    Bx�^�  
�          @��þB�\�����)����G�C�=q�B�\���>L��?�  C�J=                                    Bx�mP  
t          @��H�.{��ff�=p����C�8R�.{�񙚽��fffC�p�                                    Bx�{�  
�          @�=q��{�����$z�����C�~���{��\)>�\)@ffC�                                    Bx���  �          @񙚿�=q��ff��(��q�C��R��=q���?fff@ۅC��
                                    Bx��B  �          @�׿�G�����Q�����C��=��G���>��@e�C�q                                    Bx���  
�          @񙚿�����Q��#�
����C���������R>�z�@(�C���                                    Bx���  T          @�Q����G�����h��C�P�����
=?n{@��C�z�                                    Bx��4  
�          @�Q쿨�����Ϳ���]p�C��f�����陚?��A��C��                                     Bx���  
�          @�G������H�����ffC�S3������?&ff@�\)C���                                    Bx��  T          @��ÿ���Q��*=q���
C�������  >L��?��
C���                                    Bx��&  
�          @�G��\)��G��'����\C��q�\)��Q�>��?��HC��H                                    Bx���  T          @�׿����=q�����{C��������z�?��@�33C��=                                    Bx�r  "          @�33�#�
���Ϳ޸R�[�Cx���#�
���?k�@�\)Cyn                                    Bx�  
�          @���(���\)��Q��\)Cz��(���?��HA9Cy�f                                    Bx�+�  
�          @��H�:�H��{��z��9�Ct�q�:�H��Q�?��A�Ct��                                    Bx�:d  "          @�Q쿚�H���
���R�  C��q���H�����H�.�\C�h�                                    Bx�I
  �          @�{�ٙ��׮�tz���=qC�)�ٙ���{�u��p�C��                                    Bx�W�  n          @�
=��{����Q���z�C#׿�{�����33�!G�C�<)                                    Bx�fV  
�          @������ۅ�X����33C{�=�����z��\�k�C}��                                    Bx�t�  T          @�\)�>�R��  �B�\���Cv0��>�R���;k���Cx{                                    Bx���  "          @���aG���Q��Q�����Crs3�aG�����>Ǯ@2�\Cs�                                    Bx��H  T          A Q��n�R��(����H�D��Cq�\�n�R��Q�?��@�33Cr                                      Bx���  �          A ���]p���p��h���У�Ct@ �]p���\)?�z�A\Q�Cs��                                    Bx���  �          A ���!G��������\)C{���!G���(�@+�A�  Cz��                                    Bx��:  �          Ap���H����z�H��p�C|u���H��ff@�\Aip�C{�q                                    Bx���  "          @�ff����{�33����Cjc�����33>�  ?�ffCl)                                    Bx�ۆ  �          @����
=��p��2�\���Ck�f��
=�أ׾L�Ϳ���Cn�                                    Bx��,  T          A ���!G���33� ���hz�C{��!G����?xQ�@�p�C{�
                                    Bx���  �          A���L(���(����s33Cu�H�L(���z�?G�@���Cv��                                    Bx�x  "          A���l(����Ǯ�.=qCr���l(���G�?��Ap�Cs�                                    Bx�  �          A��n{������ (�Cr�{�n{���?�Q�A"ffCr�\                                    Bx�$�  T          A�\���H��׿z��~{Cl�\���H��ff@
�HAo
=Ck�{                                    Bx�3j  T          A�H��Q���p����xQ�CkxR��Q��Ӆ@��Ak�
Cj5�                                    Bx�B  :          A(������H��(��9��Cl�
���ָR@A33Ck�                                    Bx�P�  
t          A����(��ڏ\=���?(��Ci���(���Q�@.�RA��HCf�H                                    Bx�_\  �          AQ���  ��{?
=q@g�Cg�H��  ��
=@E�A�p�Cd��                                    Bx�n  T          A����\)��
=?��@���Ch
��\)��
=@I��A�\)Cd�q                                    Bx�|�  T          A�
������?u@�Q�Ch^���������@^{A�
=CdW
                                    Bx��N  "          Az�������?��
A
�\ChY��������@qG�AѮCc�3                                    Bx���  
�          A�H��33��
=@!�A�Q�C_\��33���@�ffA��CV�                                    Bx���  
�          A
=��\)���@^�RA�
=CW�=��\)�333@�ffB�RCL(�                                    Bx��@  �          A����p���Q�@AG�A���C^޸��p��k�@�=qB�CT�q                                    Bx���  �          A�������
=@=p�A���C`�q�����y��@�33B��CW8R                                    Bx�Ԍ  �          AQ���Q����H@QG�A���C^ٚ��Q��j=q@��HB�HCTh�                                    Bx��2  "          A���
=���@Y��A���C^}q��
=�`��@�p�Bz�CS�                                    Bx���  "          A33�����ff@G�A�(�C`33����s�
@�  Bz�CV8R                                    Bx� ~  �          A�����33@l��A�  C^������QG�@�z�B �CR�                                     Bx�$  "          A�\�������@��RB p�CW�\�����ff@�\)B,�CH�                                    Bx��  
�          A=q��������@�A�33CW�������
=@�ffB+=qCH                                    Bx�,p  "          A����z��o\)@��\B{CV����zῼ(�@�z�B>�RCC:�                                    Bx�;  �          A�\��Q��L(�@�ffBG�CP�R��Q�h��@���B7�HC=                                    Bx�I�  T          Ap���(��?\)@�(�B �\CO�q��(��(��@��
B<�RC:��                                    Bx�Xb  
�          A�R����]p�@��B�RCRW
������\@��
B0�RC@#�                                    Bx�g  
�          @����z��{@��\B"G�CL�)��zᾙ��@�z�B8�C7(�                                    Bx�u�  l          @�p���G��z=q@�p�BG�C`���G���(�@��
BH{CN�                                    Bx��T  <          @�z�@
=����  �l=qC�S3@
=�}p������$p�C�g�                                    Bx���  �          @�{@5��@  �����T��C���@5���p��z=q�
ffC���                                    Bx���  T          @�G�@J=q�1G����H�T\)C�\)@J=q��\)��=q�G�C�ٚ                                    Bx��F  
�          @���@>{�Vff���
�R�C��H@>{��(���33��
C�q�                                    Bx���  �          @�  @(���p������A  C���@(�����`  ����C�=q                                    Bx�͒  �          @�Q�@�������z��E�HC��@����  �h����=qC��                                    Bx��8  "          @��
@p���Q���p��I33C���@p������i�����C�O\                                    Bx���  
�          @�ff@   ��������?�HC��@   ���
�Y����C�
                                    Bx���  "          @�����
�\��p��	z�C�zᾣ�
��\)���H�6=qC��R                                    Bx�*  �          @�R<��
��������G�C��<��
��p���33�l��C��                                    Bx��  T          @��R�n{���������{C����n{��=q���_\)C�c�                                    Bx�%v  
�          @�=q�У����
�j�H��z�C���У����ÿJ=q��33C���                                    Bx�4  n          @�\)?�
=�:�H��Q��L
=C�R?�
=��  �#33����C�P�                                    Bx�B�  T          @��@�Q�>������R�CG�@{�@�Q�������5  C�w
                                    Bx�Qh  �          @�33@��H�$z���
=�'��C�ff@��H���R�U��C��q                                    Bx�`  
�          @��H@��
��
=�����Gp�C�w
@��
�n{��ff�{C�Ǯ                                    Bx�n�  �          @�@�Q�Ǯ��=q�E�C�p�@�Q��g���Q��C��H                                    Bx�}Z  :          @��@��
�@  ����\)C�0�@��
�����<�����C��{                                    Bx��   n          @�\)@�(��E���33�ffC�+�@�(���z��AG���(�C�w
                                    Bx���  �          @���@��=��
�����1p�?z�H@����ff��ff�!�\C��                                    Bx��L  "          @�G�@�=q?�ff��(��3A��@�=q��Q���p��AG�C��H                                    Bx���  
�          @��
@��R?}p����\�.��A+�@��R��=q����.{C�                                      Bx�Ƙ  
�          @�
=@�{>#�
��=q�6�R?�\@�{��Q���ff�'33C��H                                    Bx��>  �          @���@��
�����\)�;ffC���@��
�!���(��"  C���                                    Bx���  �          @���@����\��=q�4�HC�u�@��AG���  �\)C��                                     Bx��  �          @�Q�@�\)��z���\)�&z�C��@�\)�AG����
��\C���                                    Bx�0  �          @�{@��H�u���\�,�\C��@��H�8Q������Q�C��
                                    Bx��  �          @���@����H��  �*=qC�g�@���H����G�C��q                                    Bx�|  �          @��H@�(�����ff�*  C�,�@�(��(���33�ffC���                                    Bx�-"  �          @�
=@��׿���33�4�C��)@����!����R��C��                                     Bx�;�  �          @�Q�@3�
������
=�}z�C�O\@3�
��33��ff�9�C�u�                                    Bx�Jn  �          @��@=q�(������3C��H@=q����(��6ffC��H                                    Bx�Y  �          @�G�@p���H��{�{  C���@p���33��ff�.�HC�k�                                    Bx�g�  �          @��H@\)����ۅ8RC�W
@\)������p��7  C���                                    Bx�v`  �          @�@#�
�0����(��q��C�l�@#�
��z���Q��$Q�C�:�                                    Bx��  �          @�z�@7
=�XQ���
=�Z33C�q@7
=������z��Q�C���                                    Bx���  �          @�@*=q�U��ƸR�^�C�U�@*=q��  �������C��=                                    Bx��R  
�          @��H?�33�HQ���ff�x��C��?�33��Q����"G�C�XR                                    Bx���  �          @�\?����8Q���\)�z�C�?������������'=qC��                                    Bx���  T          @�\)?�z��Y����(��n
=C���?�z���\)��  �C�:�                                    Bx��D  �          @�p�?��P  ���
�q\)C�Ff?����H�����\C�u�                                    Bx���  �          @�(�?�z���Q������FC�y�?�z���p��a���z�C�H�                                    Bx��  �          @�R@33����HQ����C�@33��{�&ff��{C��{                                    Bx��6  �          @�=q@(���L(���Q��7p�C���@(������*=q��33C��                                    Bx��  �          @��
@O\)�p���33�a=qC�o\@O\)�����z���C�k�                                    Bx��  �          @�G�@C33�S�
��\)�T
=C�U�@C33������{�=qC���                                    Bx�&(  �          @�=q@O\)�����c33��  C��@O\)��ff��  ���
C���                                    Bx�4�  �          @���@.{����������C�s3@.{�Ӆ��=q�ep�C�+�                                    Bx�Ct  �          @�G�@>{���R��  �*C��)@>{����-p���{C��\                                    Bx�R  �          @�33@H���y����z��<ffC�s3@H����ff�Z�H�݅C�n                                    Bx�`�  T          @�ff@G
=���\�����8�C��)@G
=�����W��У�C�j=                                    Bx�of  �          @�G�@8���l(���z��M��C��@8�����R�}p���C�l�                                    Bx�~  �          @��?У����������C�RC��?У����H�S33�ծC�33                                    Bx���  �          @�\@�
�������I(�C�O\@�
���j�H����C�@                                     Bx��X  �          @�33@����p���z��K33C��=@�������qG���{C���                                    Bx���  �          @�@=q��Q���
=�O�\C���@=q��G��z=q��  C��                                    Bx���  �          @�@   �X����{�_=qC�8R@   ��������p�C��                                    Bx��J  �          @�Q�@�\�R�\����kp�C��@�\������33�p�C�
                                    Bx���  �          @�Q�?�G��9����ff  C���?�G����������)(�C��                                    Bx��  �          @�\)?�������H�\C�?�������\�6��C�7
                                    Bx��<  �          @���@33��Q�����@�C���@33��
=�J=q��G�C��3                                    Bx��  �          @��@#33��z�����z�C�0�@#33��p���
=�y�C��{                                    Bx��  �          @�@%��o\)��33�P�\C�N@%�����y����Q�C�)                                    Bx�.  �          @�{@*�H��H�Ϯ�s�C��f@*�H������Q��*  C���                                    Bx�-�  �          @���@=p���
=����zQ�C�1�@=p���(����
�:\)C�Ф                                    Bx�<z  �          @�  @,���Q�����z��C���@,�����\�����3G�C�=q                                    Bx�K   �          @�\)@,(������
=��C�s3@,(���(����:G�C��f                                    Bx�Y�  �          @���@(Q���\���
�u\)C���@(Q�������R�,G�C���                                    Bx�hl  �          @���@��Mp���\)�\�RC�8R@���ff��Q���C�<)                                    Bx�w  �          @�\@���������:\)C��=@�����B�\���C�ff                                    Bx���  �          @�p�@ff�����=q�0��C��@ff�����*�H���C��R                                    Bx��^  �          @أ�@���(���\)�133C��@���z��'���z�C��
                                    Bx��  �          @�@���33��(��${C��=@���ff�p����C�l�                                    Bx���  T          @�z�?�����n�R���C��R?����ÿ�\)�>ffC�q�                                    Bx��P  
�          @�=q?�Q����\�j�H�
=C�y�?�Q��Å�����A�C���                                    Bx���  �          @ҏ\@����z��tz���
C�R@����  �˅�_�C�\                                    Bx�ݜ  �          @�  @{��z��~{�C��f@{���\��=q���HC�j=                                    Bx��B  �          @���@{���������#p�C�H@{����  ���C��H                                    Bx���  �          @Ӆ@�R��G��}p��G�C�|)@�R���R��G��v�RC�B�                                    Bx� 	�  �          @�Q�@'���������)��C��q@'���{�!G���\)C�`                                     Bx� 4  �          @��
@�
�����(��"  C�O\@�
�����Q�����C�^�                                    Bx� &�  �          @�33?�\)���
�L�����C�f?�\)���R�n{�\)C���                                    Bx� 5�  �          @���?�������(Q�����C��3?����zᾳ33�U�C�AH                                    Bx� D&  �          @ȣ�?�G���Q��
=����C�/\?�G���(�>��
@:�HC��{                                    Bx� R�  �          @���?��
��p����R�fffC�^�?��
����?Y��A�\C�:�                                    Bx� ar  �          @Å?У�����@  ��\)C��\?У����\�E����C�Ф                                    Bx� p  �          @�p�?�����\�qG����C�'�?����p���=q�d��C�e                                    Bx� ~�  T          @Ǯ?�z��z�H����;��C���?�z���z��*=q�ɮC��{                                    Bx� �d  �          @��H?�G��z�H��ff�>=qC�3?�G���ff�2�\�ϙ�C��                                    Bx� �
  �          @��?��������\)�1��C���?��������33��
=C��                                    Bx� ��  �          @��H?��H��{�Vff�p�C�޸?��H�����33�-p�C��3                                    Bx� �V  �          @ʏ\?����(��aG��33C�@ ?����33��p��3
=C�H�                                    Bx� ��  �          @ə�?�����=q�_\)���C�N?������ÿ�(��3�C�.                                    Bx� ֢  �          @�ff?��\���H�R�\� \)C��?��\��
=���
��C��                                    Bx� �H  �          @���?��H���H�G���(�C�f?��H����aG��  C���                                    Bx� ��  �          @�ff?��H�����K���\)C�e?��H���
�u��HC��                                    Bx�!�  �          @�?У����\�G���Q�C���?У���z�aG��33C���                                    Bx�!:  �          @��?�����E��RC�"�?����
�\(�� ��C��3                                    Bx�!�  �          @\?��H�����1G���\)C��f?��H��
=�
=��G�C�y�                                    Bx�!.�  T          @��H@
=q���R�333����C��q@
=q����#�
����C�k�                                    Bx�!=,  T          @�(�@
=��G��@  ��\C��@
=���\�c�
���C�o\                                    Bx�!K�  �          @�=q@���=q�>{��=qC�1�@����H�Y��� ��C��=                                    Bx�!Zx  �          @��@
�H��(��O\)�z�C���@
�H���׿�
=�3�C���                                    Bx�!i  �          @���?�������dz���C�H?������(��]G�C�*=                                    Bx�!w�  �          @��H?�  ���\�e�  C�u�?�  ���
�\�hQ�C���                                    Bx�!�j  �          @�Q�?������n{�=qC��R?����p���  ��G�C���                                    Bx�!�  �          @�?��H��  �����+
=C��{?��H��  �	����  C�                                    Bx�!��  �          @���@"�\��G��O\)�C��@"�\�����z��b�RC�~�                                    Bx�!�\  �          @�=q@@  �u��J�H�=qC��@@  ���׿�
=�c�C�)                                    Bx�!�  �          @��@3�
�vff�Y����C��@3�
����У�����C��                                    Bx�!Ϩ  �          @���?˅��  �a���\C�<)?˅��G�������G�C�Y�                                    Bx�!�N  �          @��\?����R�e�{C��?����׿���\)C��q                                    Bx�!��  T          @��H?˅��=q�e����C��?˅���
��\)��G�C�=q                                    Bx�!��  �          @��
@33�����g
=��C�(�@33��\)���H��\)C���                                    Bx�"
@  �          @�z�@�����fff��RC�q@��33��  ����C�U�                                    Bx�"�  �          @�@�~{�mp��G�C�C�@��33�����{C�U�                                    Bx�"'�  �          @�z�@=q��  �dz��Q�C��=@=q��=q��p���33C��R                                    Bx�"62  �          @�\)@��u��}p��%�\C��=@���=q�
=q��G�C�]q                                    Bx�"D�  �          @���@����z��S33���C�O\@����33��Q��g
=C���                                    Bx�"S~  �          @��@$z����R�*=q����C��3@$z�����Q��\)C��                                    Bx�"b$  �          @��
@]p��U�:�H��(�C�f@]p����R��33�f�RC���                                    Bx�"p�  �          @��H@a��>{�H���	{C��)@a��|�Ϳ�  ��(�C��                                    Bx�"p  �          @���@]p��5��O\)��RC�]q@]p��w
=�����z�C��                                    Bx�"�  �          @���@H���L���g
=�p�C�@ @H�����H�ff����C��\                                    Bx�"��  �          @��\@K��E�qG��!\)C��H@K���������\C�4{                                    Bx�"�b  �          @���@g
=�\)�qG��"��C��3@g
=�p  �!G���(�C��3                                    Bx�"�  �          @��@@���XQ��u��!
=C��{@@����33��R����C��R                                    Bx�"Ȯ  �          @��@z�����|����C�\)@z������z�����C�                                    Bx�"�T  �          @�=q@�����H�w
=��
C�*=@����Q��p����C�^�                                    Bx�"��  �          @���?�����W
=���C��3?�����
��ff�LQ�C���                                    Bx�"��  �          @�Q�?L������H����RC��f?L�����\��=q�,��C��                                    Bx�#F  �          @�?�=q��(��W
=�\)C�k�?�=q���\���h  C���                                    Bx�#�  �          @��\?�(���\)�Z�H���C�{?�(����R��Q��d(�C���                                    Bx�# �  �          @�  ?�������^�R�33C��
?�������\�up�C�=q                                    Bx�#/8  �          @��R?����g��Q�C�c�?���  ��p�����C��{                                    Bx�#=�  �          @�
=?�=q��
=�g
=���C���?�=q���ÿ��H���C�R                                    Bx�#L�  �          @�ff?�����\)�e���C���?������ÿ�Q���C�(�                                    Bx�#[*  �          @��
?�����
�e��
C��\?������p����
C��)                                    Bx�#i�  �          @��@G��p  �|���+�C�*=@G���
=�{���C��                                    Bx�#xv  �          @�(�?�z����\�mp����C���?�z���p���\��(�C�Ff                                    Bx�#�  �          @��?�Q���p��o\)��HC�
=?�Q����ÿ�{��=qC��H                                    Bx�#��  �          @�ff@   �z=q�hQ��\)C���@   ��  ��{��z�C��                                     Bx�#�h  �          @��\?��u��g��"  C���?��������C��                                    Bx�#�  �          @���?�
=�s�
�j=q�%G�C��H?�
=��p���
=���HC�j=                                    Bx�#��  �          @�p�?�(��y���g��p�C�c�?�(������{���RC��f                                    Bx�#�Z  �          @�ff?����|(��g
=��RC�+�?�����Q������C��q                                    Bx�#�   �          @�ff@
=�s�
�j�H� ��C���@
=��p��������C���                                    Bx�#��  �          @���@%��O\)�i���%�C�:�@%���z������z�C�5�                                    Bx�#�L  �          @�\)@0  �C�
�hQ��%�\C���@0  ���R�(���Q�C���                                    Bx�$
�  
�          @�(�@ ���>�R�mp��.Q�C�3@ ����p��33��ffC��                                    Bx�$�  �          @�z�@  �Vff�dz��%�
C��@  ���R�����RC�`                                     Bx�$(>  �          @��@G��xQ��tz��$�C��
@G���G���
��Q�C���                                    Bx�$6�  �          @�=q?�33�w
=�`  ��\C�R?�33���Ϳ��
��p�C��
                                    Bx�$E�  �          @�\)@(��u�fff�Q�C���@(���p���\)��33C��                                    Bx�$T0  �          @�G�@�\�mp��q��#��C��
@�\����ff��z�C���                                    Bx�$b�  �          @�z�@\)�U��aG��%
=C��R@\)����   ��  C�u�                                    Bx�$q|  �          @�33@$z��hQ��W
=�33C��@$z���(��޸R��G�C���                                    Bx�$�"  �          @��R?������H�N�R��
C�|)?�����  �����w�C���                                    Bx�$��  �          @�p�?���u��fff�&�C�G�?����������G�C�Y�                                    Bx�$�n  �          @�33?�(��vff�a���C���?�(����Ϳ����C���                                    Bx�$�  �          @��@2�\�u��O\)�	�\C�
=@2�\��Q�Ǯ�|��C�*=                                    Bx�$��  �          @���@<���a��A����C��
@<�������p��yC��                                    Bx�$�`  �          @�ff@5��XQ��L����HC��q@5����H�ٙ�����C��
                                    Bx�$�  T          @��H@0  �X���`  ��
C���@0  ��ff��(�����C��)                                    Bx�$�  �          @��\@,(��W
=�c33��C�S3@,(���ff����Q�C��{                                    Bx�$�R  �          @��
@�\�c33�u��,ffC��R@�\��\)�{���RC���                                    Bx�%�  �          @��H@
�H�X���vff�.C�K�@
�H���H�33�¸RC���                                    Bx�%�  �          @��@���p���fff�33C�,�@�����\��
=���C�>�                                    Bx�%!D  �          @��
@���i���j=q�"�C��@����  ������C�l�                                    Bx�%/�  �          @��@p��\(��o\)�)��C�T{@p����H����G�C��
                                    Bx�%>�  �          @���@
=q�g
=�dz��!p�C�w
@
=q����(���G�C�e                                    Bx�%M6  �          @��
@�
�k��c�
�\)C��@�
�����
=����C���                                    Bx�%[�  �          @�G�@.{�[��q��#G�C�,�@.{���H��R��p�C�K�                                    Bx�%j�  �          @�p�@3�
�\���`���{C���@3�
��Q��p���33C��)                                    Bx�%y(  �          @��
@333�^{�Z=q�\)C�t{@333�����\)����C��q                                    Bx�%��  
�          @�@S�
�8���e�
=C�p�@S�
����������
C���                                    Bx�%�t  �          @�@1G��Q��b�\��C�3@1G�����z���z�C�4{                                    Bx�%�  �          @�(�@z��mp��a��C�H@z���  ��z�����C���                                    Bx�%��  �          @��?�=q��z��S�
��HC��\?�=q��=q��ff��  C��                                    Bx�%�f  �          @�z�@���tz��`���33C���@����33������33C�/\                                    Bx�%�  �          @��R@�
�\)�W
=�G�C�@�
��ff��33��=qC��f                                    Bx�%߲  �          @���@4z��J�H�aG���HC��3@4z���  �ff��p�C���                                    Bx�%�X  �          @��R@L���(Q��c�
�"C�E@L���p������HC�0�                                    Bx�%��  �          @�z�@H���!G��dz��&=qC���@H���j=q�������C�T{                                    Bx�&�  �          @�Q�@H���{�r�\�-�C��@H���l���'����C�/\                                    Bx�&J  �          @�p�@Mp��$z��xQ��-G�C���@Mp��tz��*=q���C�                                    Bx�&(�  T          @��@J=q�2�\�l���%  C�G�@J=q�|���=q��  C�Q�                                    Bx�&7�  �          @��@E�*=q�k��(ffC��q@E�u�������
C�q�                                    Bx�&F<  �          @���@K��#33�o\)�*(�C��f@K��o\)�"�\��33C�/\                                    Bx�&T�  �          @�=q@<���7
=�p���)�
C��{@<����G������
=C��                                    Bx�&c�  �          @��@B�\�3�
�vff�+��C��H@B�\�����#33���HC���                                    Bx�&r.  �          @�=q@G��*=q�p���)��C���@G��vff�!���  C���                                    Bx�&��  �          @�G�@HQ��$z��q��,
=C�O\@HQ��qG��%���(�C�ٚ                                    Bx�&�z  �          @�{@@���9���w
=�*��C�f@@������!��ӮC�q                                    Bx�&�   �          @�@*=q�Mp��u�*�C�ٚ@*=q��(��=q��\)C��H                                    Bx�&��  �          @�
=@9���A��xQ��+  C��
@9����\)� ����ffC�/\                                    Bx�&�l  �          @�(�@Dz��0  �u��+p�C�{@Dz��|���$z����C��                                    Bx�&�  �          @�p�@?\)�6ff�xQ��,�C�:�@?\)����%��33C�4{                                    Bx�&ظ  T          @��R@AG��<(��vff�)p�C��H@AG���(��!G��хC�3                                    Bx�&�^  
�          @�{@@���>�R�r�\�'z�C�� @@�������p���
=C��{                                    Bx�&�  �          @��R@<���<���x���,
=C��f@<������#�
�ՙ�C��3                                    Bx�'�  �          @�  @6ff�8Q����\�4�C�Z�@6ff����0����C�8R                                    Bx�'P  �          @���@3�
�8�����
�6p�C�
@3�
���333��ffC��3                                    Bx�'!�  �          @���@333�@  ���H�3C��H@333�����/\)�ᙚC��)                                    Bx�'0�  �          @�G�@5��A������1G�C��H@5������+�����C��
                                    Bx�'?B  �          @��\@7
=�Fff��G��/
=C�\)@7
=���H�)����ffC���                                    Bx�'M�  �          @��@C�
�<����G��.=qC��@C�
��ff�-p��ܣ�C��                                    Bx�'\�  �          @��\@C�
�5����\�1�\C��)@C�
��33�2�\���C�Y�                                    Bx�'k4  	�          @���@1��B�\�����2(�C�9�@1������,(��ޏ\C�y�                                    Bx�'y�  T          @��\@333�C�
��33�2��C�=q@333��=q�/\)��  C�xR                                    Bx�'��  
�          @��@5��9������4�C�(�@5�����0����C�!H                                    Bx�'�&  
�          @�  @8Q��*�H��ff�;�C��q@8Q��\)�>�R���C��                                    Bx�'��  T          @�(�@5��1����\�=�HC���@5���z��C33����C�,�                                    Bx�'�r  "          @�=q@8Q��5�����@�RC��{@8Q���Q��N{� {C��                                    Bx�'�  �          @\@E�(Q���Q��@�C��
@E����Q���C��=                                    Bx�'Ѿ  T          @���@>{�
�H��ff�IC��\@>{�fff�X�����C��q                                    Bx�'�d  "          @���@>{�%�����D��C�p�@>{�����U��C�(�                                    Bx�'�
  �          @��H@2�\�7����\�CQ�C�{@2�\����QG����C�xR                                    Bx�'��  T          @��
@6ff�<(������?�C��@6ff����N{����C��)                                    Bx�(V  "          @\@9���-p�����D��C�u�@9������Vff��HC�o\                                    Bx�(�  T          @�(�@;��/\)��(��Dz�C�y�@;���{�W���\C�y�                                    Bx�()�  �          @\@3�
�,(�����H�C�#�@3�
����Z�H�	�C��                                    Bx�(8H  �          @�  @��2�\��z��GQ�C���@������HQ��
=C�/\                                    Bx�(F�  �          @��@%��@�������9�C�Ff@%������5���(�C��=                                    Bx�(U�  
�          @��@(Q��:=q��=q�8=qC��@(Q������2�\����C�+�                                    Bx�(d:  
�          @��\@{�;������9Q�C��@{�����/\)��Q�C�c�                                    Bx�(r�  T          @���@���0����(��B(�C�q�@�������:=q���RC�W
                                    Bx�(��  �          @�G�@%��-p����\�>  C��3@%��}p��8Q�����C��                                    Bx�(�,  "          @��H@"�\�*=q���R�Cz�C�ٚ@"�\�}p��AG���HC�Y�                                    Bx�(��  �          @���@(���0  �tz��4{C��@(���z=q�'���RC��                                    Bx�(�x  �          @���@1��+��|���6z�C��@1��x���1G�����C��=                                    Bx�(�  
�          @��\@=p��%��~{�5Q�C�t{@=p��s33�5���C��
                                    Bx�(��  �          @��H@J=q�!G��w
=�.C���@J=q�l���0  ��C�:�                                    Bx�(�j  
�          @��@;��G��z�H�:�C�q@;��_\)�9����
C�f                                    Bx�(�  �          @���@)���33��33�J�C�&f@)���U�H���z�C�:�                                    Bx�(��  	�          @���@!����~{�D33C���@!��c33�:�H�33C��                                     Bx�)\  �          @�\)@�R�1��tz��<�C�` @�R�z�H�(����=qC���                                    Bx�)  �          @��
@ff�5�l���9\)C�>�@ff�{��   ��{C��                                    Bx�)"�  �          @�z�?��H�J=q�c33�.��C��?��H��{�����=qC��                                    Bx�)1N  T          @�(�@��P���k��,C��H@���=q���z�C�`                                     Bx�)?�  
�          @��
@���L���n{�/33C�AH@�����������z�C���                                    Bx�)N�  <          @�z�?�33�b�\�K��\)C��?�33������
��z�C��H                                    Bx�)]@  
�          @���@   �J�H�Vff�'
=C�"�@   ���
�z���=qC��3                                    Bx�)k�  
�          @�Q�?��R�G��X���)�C�G�?��R���H�Q���z�C���                                    Bx�)z�  
�          @���@�\�AG��Tz��$�C��R@�\�~{�ff����C�                                      Bx�)�2  	�          @��
@  �4z��hQ��4�C�G�@  �x���p��噚C��                                    Bx�)��  "          @���@33�5�h���3�C�� @33�y���p���z�C�>�                                    Bx�)�~  	�          @�(�@{�333�j�H�7
=C�33@{�xQ�� ����z�C��f                                    Bx�)�$  "          @��@��0���h���5�RC��@��tz��\)��C�k�                                    Bx�)��  
�          @�=q@  �(Q��mp��;�
C�Ff@  �n�R�&ff���C���                                    Bx�)�p  	z          @�=q@���\�tz��D��C�H@��\(��5��p�C�H�                                    Bx�)�  "          @��\@�����{��K\)C�O\@��X���>{�(�C�:�                                    Bx�)�  
�          @�G�@
=q�!��q��B�
C�<)@
=q�i���.{���C�J=                                    Bx�)�b  �          @���@z��   �w
=�G�C�Ǯ@z��i���333�{C��H                                    Bx�*  
�          @���@
=����u�G�RC�\)@
=�e�333�C�4{                                    Bx�*�  
�          @�z�@���R�j=q�A��C�  @��c�
�'���
C�(�                                    Bx�**T  
�          @��@33��R�l���C��C�˅@33�dz��*=q�Q�C��3                                    Bx�*8�  �          @�{@���#�
�h���=�C��@���g��%���{C�B�                                    Bx�*G�  
�          @�p�@��� ���e��:�\C���@���c�
�"�\����C�4{                                    Bx�*VF  "          @��@���&ff�g
=�933C�~�@���i���"�\��
=C���                                    Bx�*d�  
�          @�Q�@\)�+��e�733C���@\)�mp��   ��  C���                                    Bx�*s�  
Z          @�G�@��5�c�
�3\)C��)@��vff��H��p�C���                                    Bx�*�8  "          @�
=@��;��Z=q�-\)C��@��x���  ���C�Ff                                    Bx�*��  "          @�33?�\)�:�H�X���1=qC�K�?�\)�xQ��\)���C��                                     Bx�*��  
�          @�=q?��>{�U�/Q�C��\?��y��������C�s3                                    Bx�*�*  �          @��?����?\)�W��/��C���?����{������C�l�                                    Bx�*��  
�          @��?�
=�=p��Y���/=qC��?�
=�z=q�\)��\)C�&f                                    Bx�*�v  "          @�33?��
�G
=�P���)
=C��q?��
��Q���
�ɮC��R                                    Bx�*�  T          @��
?�
=�K��QG��)33C��H?�
=���\��
��=qC�,�                                    Bx�*��  �          @�
=?����^�R�fff�.ffC�U�?������R�G���=qC�f                                    Bx�*�h  
�          @��H?�  �h���g
=�)�C�,�?�  ���
��R��\)C��)                                    Bx�+  �          @�=q?��
�c33�i���-�C���?��
��G���
�͙�C�K�                                    Bx�+�  �          @��?�(��W��g��.�C���?�(�������  C�                                    Bx�+#Z  
Z          @�(�?�Q��b�\�l(��,�HC��=?�Q���G��ff����C�:�                                    Bx�+2             @��R?�{�l(��l(��*�C�?�{����
���C�w
                                    Bx�+@�  
�          @�ff?�  �z�H�dz��#C��)?�  �������G�C�9�                                    Bx�+OL  
�          @���?�ff�n�R�e��&(�C�P�?�ff���(����
C�0�                                    Bx�+]�  "          @�p�?�=q�n{�fff�&��C��?�=q���{��\)C�XR                                    Bx�+l�  �          @�ff?�{�j=q�l(��*�HC�ٚ?�{�������ʣ�C���                                    Bx�+{>  
�          @��?޸R�^�R�l���.Q�C�C�?޸R��\)�����=qC���                                    Bx�+��  :          @��?���Tz��r�\�4�C�,�?������"�\��z�C�/\                                    Bx�+��  
          @��@��^�R�q��+��C��\@���  ��R��ffC��q                                    Bx�+�0  �          @�(�@
=�e��s33�)��C�C�@
=���H�{��(�C�S3                                    Bx�+��  "          @��?�Q��n{�q��(
=C��=?�Q���
=�=q��  C�,�                                    Bx�+�|  
�          @��H@(��o\)�z=q�(ffC�*=@(������!��ͮC�H�                                    Bx�+�"  T          @�=q@(��n�R����*C��3@(����H�.�R��C�^�                                    Bx�+��  �          @�@4z��p����Q��!�HC�o\@4z����\�'����C�"�                                    Bx�+�n  l          @�
=@1��q���=q�#�C�,�@1�����+��̸RC�޸                                    Bx�+�  
�          @�ff@%�hQ��{��%�C���@%���&ff��\)C�y�                                    Bx�,�  "          @�@p��e������+�C�E@p�����,�����C��                                    Bx�,`  
�          @�\)@(��x���tz��  C��@(���z���H���HC�C�                                    Bx�,+  f          @���@(Q��|(��{����C��q@(Q���
=� ����Q�C��{                                    Bx�,9�  �          @���@%�vff�w��(�C���@%�����R��C��                                    Bx�,HR  �          @�Q�@!G��o\)�{��$p�C��R@!G������%���z�C��                                    Bx�,V�  �          @�Q�@)���g
=�\)�&�C��@)����p��*�H���C�Ǯ                                    Bx�,e�  �          @�33@����s33��C���@��(�����\)C�B�                                    Bx�,tD  �          @��H@���n�R��  �1�
C�C�@������9����  C�.                                    Bx�,��  �          @��H@%��i����(��*��C��f@%�����3�
��  C�J=                                    Bx�,��  �          @\@'
=�n{�����&ffC�~�@'
=�����,������C�K�                                    Bx�,�6  T          @�33@,(��mp���Q��$��C���@,(���Q��,(���G�C��q                                    Bx�,��  �          @�(�@7
=�r�\�xQ��(�C��=@7
=��G��"�\��  C�l�                                    Bx�,��  �          @�z�@5�xQ��s33��C��@5��33�(���  C�*=                                    Bx�,�(  T          @�(�@;��u��q��z�C��
@;���G��(����
C���                                    Bx�,��  �          @���@;��s33�u��C���@;������   ��ffC�                                    Bx�,�t  �          @�(�@,(��hQ����
�)�C�>�@,(����R�5��܏\C���                                    Bx�,�  �          @���@!��u�����%ffC���@!���z��,����33C��3                                    Bx�-�  �          @�p�@-p��j=q��p��)��C�C�@-p�����7���{C�޸                                    Bx�-f  �          @�(�@$z��hQ����R�-�
C���@$z���\)�;���z�C�@                                     Bx�-$  �          @�@,(��q����\�%G�C��\@,(����\�0����{C���                                    Bx�-2�  �          @�ff@7��s33�}p���C���@7�����(�����C�j=                                    Bx�-AX  �          @�(�@1��u�w��C��R@1���=q�"�\�ď\C�H                                    Bx�-O�  �          @��@5�p���|����C���@5��Q��)����z�C�e                                    Bx�-^�  �          @�@:�H�b�\�����(�\C���@:�H����:=q��RC�'�                                    Bx�-mJ  �          @�
=@$z��qG���
=�+(�C�!H@$z���33�:�H���
C���                                    Bx�-{�  T          @Ǯ@ ���n{���\�0  C��@ �����\�B�\��p�C���                                    Bx�-��  �          @ƸR@,(��]p���p��5
=C��@,(�����L�����
C�                                      Bx�-�<  �          @�\)@+��c�
����1z�C�� @+����G���=qC��                                    Bx�-��  �          @ƸR@1��]p����H�1�C�g�@1����\�H������C��H                                    Bx�-��  T          @�\)@%��`  ��ff�6�RC�8R@%���z��N�R��z�C���                                    Bx�-�.  �          @ƸR@\)�W
=���\�>p�C�N@\)�����Y���  C�^�                                    Bx�-��  T          @�@(���Z=q����6��C��
@(�������N�R��Q�C�
=                                    Bx�-�z  �          @�p�@,(��W�����6�RC�N@,(���  �O\)���
C�j=                                    Bx�-�   �          @�p�@!��^�R��p��6��C��q@!�����N{���C�]q                                    Bx�-��  �          @��
@Q��XQ���Q��>
=C��R@Q���G��U�p�C��R                                    Bx�.l  
�          @Å?�(��Q������M
=C�z�?�(���Q��g��33C��f                                    Bx�.  �          @��
?�\)�U��  �LQ�C���?�\)��=q�e���C�'�                                    Bx�.+�  �          @�33?޸R�O\)��33�SG�C�#�?޸R��  �n{�z�C��R                                    Bx�.:^  �          @\?Ǯ�Vff��=q�R=qC���?Ǯ���H�j=q�\)C�j=                                    Bx�.I  �          @\?���e�����N��C�Ф?�������b�\��RC��\                                    Bx�.W�  �          @�=q?��q����\�D�
C��R?����Tz���HC���                                    Bx�.fP  T          @���?z�H�\)���<�RC��?z�H���\�G
=��Q�C�p�                                    Bx�.t�  T          @���?��\�vff��
=�?�C�&f?��\���R�L�����C�33                                    Bx�.��  �          @���?����b�\��(��H{C���?������R�\(��
�RC�@                                     Bx�.�B  �          @��H@33�c33����>\)C�f@33��p��S�
�C��q                                    Bx�.��  T          @�(�@33�c�
�����?�
C���@33��{�W��Q�C��\                                    Bx�.��  �          @��H@�\�j�H�����9Q�C��@�\��Q��L(���G�C��R                                    Bx�.�4  T          @��H@
=�fff��{�;�C�8R@
=��ff�P���33C��                                    Bx�.��  �          @��
@�
�e��  �=��C��f@�
���R�Tz���C��                                    Bx�.ۀ  �          @���@G��dz���33�:�\C�@G���z��K�� �\C��f                                    Bx�.�&  �          @��R���n{�S�
�%Q�C�������  �	�����HC�e                                    Bx�.��  �          @�G���
=�S�
���R���RCZ)��
=�n�R�}p��   C]z�                                    Bx�/r  �          @��������HQ�0����33CR
�����L��>L��?��CR��                                    Bx�/  T          @�Q���G��U��k��z�CT+���G��P��?333@�CS�f                                    Bx�/$�  �          @Ǯ����Vff�������CT������W�>Ǯ@g
=CT�q                                    Bx�/3d  �          @�����
=�N{���
�AG�CS�3��
=�L(�?�@�33CSk�                                    Bx�/B
  �          @�=q�����E�����(�CQ!H�����@  ?5@θRCP}q                                    Bx�/P�  �          @��H��G��!�>B�\?���CL����G��Q�?aG�A(�CKB�                                    Bx�/_V  �          @����\)�!G�?��@�CL����\)���?�G�A@Q�CJp�                                    Bx�/m�  �          @�
=����ff>�(�@�Q�CJG�������?��Az�CH�                                     Bx�/|�  �          @�(�����(�>�  @�RCH
�����\?Tz�@�\)CF��                                    Bx�/�H  �          @�z���Q��
=q>k�@z�CGǮ��Q��G�?L��@�\)CF��                                    Bx�/��  �          @��H�����
>��?���CG  �������?333@ʏ\CF                                      Bx�/��  �          @�p�����
==��
?5CG+����� ��?#�
@��CFT{                                    Bx�/�:  �          @�p��\��\=��
?8Q�CF���\��Q�?�R@��CE��                                    Bx�/��  �          @��
��G�� �׼#�
��\)CFk���G���Q�?�@���CE��                                    Bx�/Ԇ  �          @�{���H�>.{?��RCF����H��(�?8Q�@�z�CE�                                    Bx�/�,  �          @������
=q    =#�
CG��������?z�@���CF��                                    Bx�/��  �          @������R��
�����c33CI(����R��>k�@�CIY�                                    Bx�0 x  �          @�z���{�zᾞ�R�3�
CIW
��{�z�>��R@1�CIW
                                    Bx�0  �          @�(���  ��þ���
=CG����  ���>��R@2�\CG��                                    Bx�0�  �          @˅��{�p���G��{�CHn��{�  >#�
?�Q�CH�q                                    Bx�0,j  T          @���
=�z�������CI0���
=�Q�=��
?+�CI�R                                    Bx�0;  
�          @�������!G���CIk����=q�#�
�L��CJ!H                                    Bx�0I�  �          @˅���H��H�+����HCJu����H� �׼��
�aG�CK8R                                    Bx�0X\  �          @��
�����"�\�p����CK�=�����,�;�  ��RCM{                                    Bx�0g  �          @����
�\)�E��ۅCJ�����
�&ff���ͿaG�CK��                                    Bx�0u�  �          @�\)����%��.{���CK�\����*=q<#�
=��
CLE                                    Bx�0�N  �          @�
=��p��#�
��R���RCK\)��p��(Q�=u?
=qCK�3                                    Bx�0��  �          @�{����� �׿�R����CK������%�=#�
>\CK��                                    Bx�0��  �          @����33�!G��8Q��θRCKL���33�'��L�;��HCL!H                                    Bx�0�@  T          @�ff��(��#33�#�
���CKu���(��(Q�<�>uCL)                                    Bx�0��  �          @�\)����%��&ff��G�CK������*=q<��
>B�\CL5�                                    Bx�0͌  �          @�
=��G���\��p��Q�CH�=��G���
>aG�?��HCH�3                                    Bx�0�2  �          @�ff�����  ��(��r�\CH}q������\>��?���CHǮ                                    Bx�0��  �          @�{���p������CJ����� ��=�?�ffCJ�R                                    Bx�0�~  T          @�{�����   ����=qCJ�3�����#�
=��
?+�CKxR                                    Bx�1$  �          @����=q�(Q�\)��  CLG���=q�+�=�G�?��
CL�q                                    Bx�1�  �          @��H����(�þ��H���CL� ����+�>B�\?�Q�CM\                                    Bx�1%p  �          @ʏ\��  �%��\)��=qCL8R��  �(��=���?h��CL�3                                    Bx�14  �          @ƸR���R�z�=p���33CJ����R���\)��=qCK�                                    Bx�1B�  �          @�\)����"�\��R��\)CL33����'�<��
>aG�CL�{                                    Bx�1Qb  �          @�
=�����!녿����G�CL(������&ff=#�
>\CL�                                     Bx�1`  �          @�
=���ff�G���G�CKaH���{�8Q��G�CLp�                                    Bx�1n�  �          @��
��{�(����H����CI  ��{�\)=u?��CIu�                                    Bx�1}T  �          @�����ff��R�0�����CI^���ff������33CJ=q                                    Bx�1��  �          @�p���G��G��#�
��  CG33��G��
=�\)��G�CH
=                                    Bx�1��  
�          @�z������R�E���
=CI������ff�W
=��Q�CJ�{                                    Bx�1�F  �          @�=q���\��׿G�����CJ����\�Q�W
=� ��CK�                                    Bx�1��  �          @��������
�Tz�����CJE������;�  �CKh�                                    Bx�1ƒ  �          @�ff��ff�33�W
=���CI���ff������=qCK{                                    Bx�1�8  �          @�\)����{�\(���z�CK������&ff�u�p�CL�3                                    Bx�1��  �          @�
=��������z�H�p�CJ�q�����#�
�\�a�CLc�                                    Bx�1�  �          @ƸR��33�(����
���CK�=��33�'���
=�xQ�CM�                                    Bx�2*  �          @�p���z��Q�#�
���RCJ�f��z��{�u��CK�)                                    Bx�2�  �          @���Q��6ff�!G��îCR����Q��W����
���
CWB�                                    Bx�2v  �          @����Fff�2�\�z=q�,��C^�Fff�hQ��I�����Ce��                                    Bx�2-  �          @����^{����HQ��Q�CV�=�^{�C�
��R�ᙚC]\)                                    Bx�2;�  �          @�=q���R�녾�z��0  CG�\���R��\>B�\?�ffCG��                                    Bx�2Jh  T          @�{��p��녿333����CI����p��Q������CJǮ                                    Bx�2Y  T          @ʏ\����
=��
=�,  CJT{����$z�����CL�                                    Bx�2g�  �          @ʏ\��Q��\)�Tz���G�CK\)��Q��'
=�k��
=CLc�                                    Bx�2vZ  �          @˅��G��(��u��CJ����G��&ff��p��XQ�CL0�                                    Bx�2�   �          @�p���������G����CI�{�����   ��ff��  CK                                      Bx�2��  �          @������
�ff�xQ���CI�)���
� �׾����hQ�CK.                                    Bx�2�L  �          @�������ÿu�
ffCJ&f����"�\�Ǯ�aG�CKs3                                    Bx�2��  �          @��H������p����CJ+�����!G���p��W�CKn                                    Bx�2��  �          @˅���H�z�u��CI�f���H��R����n�RCJ��                                    Bx�2�>  �          @�p����z�O\)��\)CIY����(�����ffCJ^�                                    Bx�2��  �          @�ff���R�G��k���HCH޸���R��H�\�X��CJ�                                    Bx�2�  �          @���ff�녿fff� ��CI  ��ff����Q��P  CJ33                                    Bx�2�0  T          @����ff�\)�Tz���RCH����ff�����R�1�CI��                                    Bx�3�  �          @�����
��ÿxQ��
�RCJ&f���
�"�\����j=qCKp�                                    Bx�3|  T          @������z�fff� ��CIu�����p���Q��Mp�CJ�H                                    Bx�3&"  �          @�(�����  �Y������CH�)����Q쾨���=p�CI�R                                    Bx�34�  �          @�p���
=�(��W
=���CH&f��
=�zᾮ{�B�\CIB�                                    Bx�3Cn  �          @����ff��Ϳc�
���RCHO\��ff��\�Z=qCI�                                     Bx�3R  �          @�ff����
�H��G��(�CG�������\��G�CIW
                                    Bx�3`�  �          @θR��  �����\���CG�R��  �ff�����HCIc�                                    Bx�3o`  �          @�\)�����
�H�z�H�\)CG�=���������H����CI#�                                    Bx�3~  �          @�
=��  �p���G��  CH:���  �Q��\����CI��                                    Bx�3��  �          @�  ��G��(���G���HCG����G��
=��\��Q�CIW
                                    Bx�3�R  �          @љ��\�\)�s33��CH:��\��þ�ff�{�CI}q                                    Bx�3��  �          @�Q���G��p��z�H�
�RCH#���G�������Q�CIu�                                    Bx�3��  �          @У������\)�n{��
CHB������Q��G��u�CI}q                                    Bx�3�D  �          @Ϯ�����(��z�H�
=qCG������ff���H���CI@                                     Bx�3��  T          @�{��
=�(��}p���CH���
=�ff�   ���RCIu�                                    Bx�3�  �          @˅�����(��c�
� z�CHff�����������i��CI��                                    Bx�3�6  �          @��
����p��Tz���Q�CH�������33�HQ�CI�{                                    Bx�4�  �          @˅��p���Ϳ:�H��(�CHc���p���
�����
CIG�                                    Bx�4�  �          @�33���
�녿O\)��\CI@ ���
������
�6ffCJ@                                     Bx�4(  �          @�����R�G��&ff��G�CH�H���R�
=�#�
��CI�{                                    Bx�4-�  �          @θR�����  �
=��\)CH����������G��uCI�                                    Bx�4<t  �          @�ff��G��
=q��R���CG���G��\)�#�
��33CH\)                                    Bx�4K  �          @θR����	���!G���=qCG� �����R�.{�\CH0�                                    Bx�4Y�  �          @θR���H�������CG.���H�
�H����z�CG��                                    Bx�4hf  �          @�  ��z��z��(��r�\CF�H��z���    <��
CG                                      Bx�4w  �          @�����R�녾�G��}p�CF�����R���#�
��CG8R                                    Bx�4��  �          @ə���\)�����Ǯ�b�\CF��\)��p�<#�
=�CFY�                                    Bx�4�X  �          @�G���{�   ���H��\)CF�{��{��
���
�5CG{                                    Bx�4��  �          @�{���\�   �����\CF����\��
��G���  CGz�                                    Bx�4��  �          @�p����� �׿
=���CG{�����8Q���CG                                    Bx�4�J  �          @������ �׿&ff����CG
�����ff�u�(�CG�H                                    Bx�4��  �          @�(����׿��R�(���\)CG
=�����z�L�Ϳ�z�CG                                    Bx�4ݖ  �          @�{�����\��R��Q�CGQ�������W
=��33CH
=                                    Bx�4�<  �          @�Q���33�ff�G���CG���33�p���33�N{CH�q                                    Bx�4��  �          @�  ���\�Q�8Q����CH
=���\��R��z��+�CH��                                    Bx�5	�  T          @ʏ\��z����E���ffCHE��z��녾��
�<(�CI0�                                    Bx�5.  �          @ʏ\���H��
�G���=qCI�=���H�=q���R�5CJs3                                    Bx�5&�  �          @��
��(���\�L����RCIL���(��=q��{�B�\CJ=q                                    Bx�55z  T          @�(���{�{�.{�ÅCHs3��{�33�u�Q�CI5�                                    Bx�5D   �          @�p���=q�G���\����CFc���=q����G��xQ�CF�f                                    Bx�5R�  �          @�ff��z��
=��
=�mp�CEu���z��p��#�
��33CE�{                                    Bx�5al  �          @θR��33��
�����CF�f��33������33CG0�                                    Bx�5p  �          @θR�����p��!G����
CH�������\�L�Ϳ�\CH�=                                    Bx�5~�  �          @θR��=q��ÿ����RCGc���=q��ͽ�G���G�CG��                                    Bx�5�^  �          @����  �
=q������HCG��  ��R�8Q�˅CHc�                                    Bx�5�  �          @�{��  �p��5�˅CH33��  �33��\)�!�CH�q                                    Bx�5��  �          @�{��
=�녿J=q��Q�CH���
=��þ�{�C�
CI�\                                    Bx�5�P  �          @θR��\)�녿k��\)CH�H��\)�=q������CI��                                    Bx�5��  �          @�{���R�G��n{�G�CH�����R����   ���CI�                                    Bx�5֜  �          @�ff���R��\�c�
��
=CI����R��H��ff����CJ
                                    Bx�5�B  �          @�ff���R��
�W
=��ffCI(����R��H�Ǯ�`��CJ)                                    Bx�5��  �          @θR��
=�녿p���ffCH���
=��H��\��Q�CJ�                                    Bx�6�  �          @Ϯ��\)��
��  �G�CI���\)��Ϳ\)���CJO\                                    Bx�64  �          @Ϯ�����\)�h��� ��CHn������������CI}q                                    Bx�6�  �          @�ff��
=�G��k���\CH�)��
=������H���\CI�                                    Bx�6.�  �          @�Q������z�Y����RCI������(�����g�CJ
=                                    Bx�6=&  �          @Ϯ��G���Ϳfff���RCH���G������H���CI�                                    Bx�6K�  �          @�  ��Q���\�p����CH�{��Q���H��\����CI�                                    Bx�6Zr  �          @�{�����H�^�R����CJE����"�\��(��q�CK=q                                    Bx�6i  �          @����H�!녿k��
=CK^����H�)����ff��G�CL^�                                    Bx�6w�  �          @�{���
��R�k��
=CJ޸���
�&ff����(�CK��                                    Bx�6�d  �          @�����33��ͿW
=���CJ��33�#�
�����b�\CK��                                    Bx�6�
  �          @���������Q���CJ�=����"�\�\�Y��CKk�                                    Bx�6��  �          @���(���ͿL�����CJ����(��#33��Q��Mp�CKs3                                    Bx�6�V  T          @�ff��p��(��=p����
CJff��p��!녾��R�.�RCK&f                                    Bx�6��  �          @�{�����p��E����
CJ�
�����#33�����=p�CKc�                                    Bx�6Ϣ  �          @����
�   �Q����CK����
�&ff��p��U�CK�f                                    Bx�6�H  �          @�{���
�   �Y�����
CK����
�&ff����j=qCK�                                    Bx�6��  �          @�p���33�\)�aG���z�CK
��33�'
=��G��|(�CL�                                    Bx�6��  �          @�{��=q�#�
�u�	G�CK��=q�,(���\����CL�=                                    Bx�7
:  �          @�������#�
���
�Q�CK�������,�Ϳz����RCM
=                                    Bx�7�  �          @�������(�ÿ}p��
=CL������1G�������CM��                                    Bx�7'�  �          @�z���Q��#�
����G�CK����Q��,�Ϳ
=����CM�                                    Bx�76,  �          @�=q��ff�!G����\�  CK�H��ff�*=q�z���  CM                                      Bx�7D�  T          @��
��  �#�
�xQ���
CL���  �,(�����\)CM�                                    Bx�7Sx  T          @��
����$z�u�33CL#�����,�Ϳ���ffCM(�                                    Bx�7b  
�          @�(�������������#�
CJ}q�����#�
�8Q���\)CKǮ                                    Bx�7p�  �          @�p����\����  �3�
CJ����\�"�\�W
=��CK��                                    Bx�7j  T          @У�����*�H��G��=qCL}q����333�\)��(�CM�                                    Bx�7�  �          @������1녿z�H���CM\)����9����\��ffCNQ�                                    Bx�7��  �          @љ�����.�R��  �z�CL�3����7
=�
=q���CM��                                    Bx�7�\  �          @�33���,�Ϳ��
�Q�CL� ���5��z���=qCM�=                                    Bx�7�  �          @љ���z��*�H���
�(�CLc���z��333�z����HCMn                                    Bx�7Ȩ  �          @�Q����H�,(��}p���CL���H�4z�
=q��Q�CM                                    Bx�7�N  �          @љ�����,�Ϳ�ff�Q�CL����5�(����\CM�{                                    Bx�7��  �          @��
��ff�*�H����{CL8R��ff�4z�333����CMaH                                    Bx�7��  �          @��
��ff�)����z��!�CL��ff�333�8Q��ȣ�CM8R                                    Bx�8@  
�          @�z�����'
=��33��CK������0�׿8Q���  CL�                                     Bx�8�  T          @�{�����*�H������CK������4z�333����CM�                                    Bx�8 �  �          @ָR�����.�R����CL^������7��&ff���CMs3                                    Bx�8/2  �          @�����  �(�ÿ�{�p�CK��  �1녿.{����CL޸                                    Bx�8=�  �          @����  �*�H��=q���CK����  �333�&ff��33CM                                    Bx�8L~  �          @��
��\)�'���{�ffCK����\)�0�׿0����  CL��                                    Bx�8[$  �          @��
���R�(�ÿ�{�
=CK�f���R�1녿0������CM                                    Bx�8i�  �          @Ӆ��{�(�ÿ����'
=CK���{�2�\�G��أ�CM&f                                    Bx�8xp  �          @�(���{�)����  �-��CL\��{�3�
�Tz���CMT{                                    Bx�8�  �          @�z���
=�(Q쿝p��*�\CK�=��
=�2�\�O\)��G�CM�                                    Bx�8��  �          @ҏ\��
=�   ��Q��&�HCJ��
=�*=q�L����
=CK��                                    Bx�8�b  �          @�=q��{�   ���\�1CJ�
��{�*=q�aG����CL(�                                    Bx�8�  �          @����\)��Ϳ����
=CJB���\)�%�@  ���HCKh�                                    Bx�8��  �          @ҏ\��G��������G�CI�3��G��!녿+����HCJ�q                                    Bx�8�T  T          @�=q������ÿ�=q��HCI�������!녿5��ffCJǮ                                    Bx�8��  �          @ҏ\�����p��xQ��Q�CJ0������$z�����  CK#�                                    Bx�8��  �          @Ӆ�\�=q�p����CI�f�\�!G��z���G�CJ�\                                    Bx�8�F  �          @�����\)�\)�J=q�޸RCJ�{��\)�%���
=�j�HCKL�                                    Bx�9
�  �          @���������H�E���Q�CI������ �׾���e�CJ��                                    Bx�9�  �          @�(���z����+����CI����z���;��
�5�CJ�\                                    Bx�9(8  �          @ə�����Q����Q�CJB�������.{�˅CJ�                                    Bx�96�  �          @�  �������z���33CI�3������þu�{CJs3                                    Bx�9E�  �          @˅��=q��H�Q���{CJ�)��=q�!G������
=CKc�                                    Bx�9T*  �          @��
��Q��   �����CKz���Q��(Q�8Q���\)CL�\                                    Bx�9b�  �          @��
����,�Ϳ\�\��CM�����8�ÿ�\)�"=qCO}q                                    Bx�9qv  �          @�(����\�(Q�˅�h  CM=q���\�5����H�.�HCN�H                                    Bx�9�  �          @�����	�����
�#�
CG�f���Q�>�\)@%�CG�q                                    Bx�9��  �          @��H��p���׼#�
��CH���p��\)>���@0  CH                                    Bx�9�h  �          @��H��p��G��\)���\CH�3��p����>8Q�?˅CH�                                    Bx�9�  �          @�(����ff�u�
=CI�
���
==�Q�?Tz�CI�                                    Bx�9��  �          @������H�k���
CJ5�����=�G�?p��CJG�                                    Bx�9�Z  �          @�p������\)�����*=qCJ�f����� ��=L��>�
=CK�                                    Bx�9�   �          @�(���33� �׾�G��z�HCKB���33�#33��Q�L��CK��                                    Bx�9�  �          @�z������(Q�   ���CL^������*�H�\)���RCL�R                                    Bx�9�L  �          @�(������(Q������
CLxR�����,(��u��CL�                                    Bx�:�  �          @�������'
=�����\CL5�����*�H�W
=��\)CL��                                    Bx�:�  �          @�z����
��Ϳ�����RCJ�3���
� �׾aG����HCK!H                                    Bx�:!>  �          @�(����\�!G������z�CKaH���\�%�����z�CK�)                                    Bx�:/�  T          @�{��(��   �G���\)CJ�q��(��%���G��}p�CK��                                    Bx�:>�  T          @�z���{��\������CI{��{�ff��z��&ffCI��                                    Bx�:M0  �          @�z���Q��
=�
=����CGQ���Q��
=q���R�/\)CG��                                    Bx�:[�  �          @�z������\�h���33CF�������ÿ#�
���RCG��                                    Bx�:j|  �          @�z����
��\��ff�\)CIW
���
�=q�=p���p�CJT{                                    Bx�:y"  �          @�p���=q�'
=�@  �ָRCL+���=q�,(������g�CLǮ                                    Bx�:��  �          @�����  �*�H�Y����(�CL���  �0�׿   ���RCM��                                    Bx�:�n  �          @�(���G��7
=�����?�COG���G��@  �p���Q�CPu�                                    Bx�:�  �          @˅�����9����G��6=qCO�3�����B�\�aG����CP��                                    Bx�:��  �          @�(������7
=����B�RCO\)�����@�׿xQ���
CP��                                    Bx�:�`  �          @�=q����/\)���R�[
=CN������:=q����%��CO�                                    Bx�:�  �          @ʏ\��Q��3�
�����@��CO{��Q��=p��u�
=CP@                                     Bx�:߬  �          @��H����;���Q��,��CP
����C�
�O\)���HCQ)                                    Bx�:�R  �          @�p����R�+���
=�(��CM)���R�333�Tz���ffCN#�                                    Bx�:��  T          @�p����R�+���p��0(�CM&f���R�3�
�aG����CN:�                                    Bx�;�  �          @�=q����*=q��(��2=qCM\)����2�\�aG�� z�CNp�                                    Bx�;D  �          @�Q���33�>{��z��HQ�CO���33�G����
�{CQ�                                    Bx�;(�  �          @Ӆ��p��A녿Ǯ�Y�CP���p��L(����#�CQh�                                    Bx�;7�  �          @�33��(��E�����W\)CP����(��P  ��33� z�CQ��                                    Bx�;F6  �          @ҏ\�����B�\���H�M��CPO\�����L�Ϳ�=q��CQ�                                    Bx�;T�  �          @ҏ\��33�G
=��G��T  CQ��33�QG���\)���CR@                                     Bx�;c�  �          @ҏ\���
�Dz��G��S�
CP�H���
�N�R�����CQ޸                                    Bx�;r(  �          @��H����E��˅�_
=CP�=����P  ���H�(��CR
                                    Bx�;��  �          @�G���Q��G�����h��CQ����Q��R�\��G��1CR�f                                    Bx�;�t  �          @�G������Fff�У��g�CQL������QG���G��1�CR��                                    Bx�;�  T          @������\�>�R�У��f�\CP����\�I����G��2ffCQk�                                    Bx�;��  �          @У���=q�:�H���H�s�CO����=q�Fff��{�@Q�CQ�                                    Bx�;�f  �          @������
�7
=���H�r�\CN�3���
�B�\��{�@��CPc�                                    Bx�;�  �          @У���33�6ff��(��tQ�CN�3��33�A녿�\)�B�RCPff                                    Bx�;ز  �          @У���=q�;���
=�o�
CO����=q�Fff��=q�=�CQ�                                    Bx�;�X  �          @�Q������;���Q��p��CO�{�����G
=����>=qCQ:�                                    Bx�;��  �          @�  �����<(���\�}p�CP\�����HQ쿷
=�J�\CQ��                                    Bx�<�  �          @�Q���ff�AG���\)����CP�R��ff�Mp���G��W33CR�                                     Bx�<J  �          @�  ��p��Fff������  CQ��p��R�\���H�O
=CS:�                                    Bx�<!�  �          @�G���
=�Dz������CQQ���
=�P�׿�p��Qp�CR�=                                    Bx�<0�  �          @Ϯ��(��>{����p�CP���(��L(���p��w\)CR�f                                    Bx�<?<  �          @������>{� ����{CQ33����J�H��z��p��CR�)                                    Bx�<M�  �          @�=q��  �<�Ϳ�Q�����CQT{��  �I�������j=qCR�3                                    Bx�<\�  
�          @�z���G��B�\��z����CQ�H��G��N�R�Ǯ�b=qCSn                                    Bx�<k.  T          @�(������A녿�{��CQǮ�����N{�\�\  CSG�                                    Bx�<y�  �          @�33�����<(�������33CQ
�����HQ��\)�l(�CR��                                    Bx�<�z  �          @�(���G��B�\��{��p�CQ����G��N�R��G��[�CSc�                                    Bx�<�   �          @˅��Q��A녿�����\CQ�3��Q��N{��ff�b{CSxR                                    Bx�<��  �          @ə���
=�A녿�ff����CR{��
=�L�Ϳ��H�V�RCS��                                    Bx�<�l  �          @ə����C33������HCR� ���N�R����b=qCS�q                                    Bx�<�  �          @�����R�C�
����
=CRff���R�N�R��(��W
=CS�\                                    Bx�<Ѹ  �          @��
��G��@  �����(�CQ����G��K��Ǯ�b�HCS                                    Bx�<�^  �          @�z����
�7
=���H��z�CP����
�C33����m�CQ�{                                    Bx�<�  �          @������;������CP�H����G
=�����h  CR&f                                    Bx�<��  �          @θR����H���z���CR�3����U��p��x��CT��                                    Bx�=P  T          @�����p��I�����(�CSc���p��Vff�޸R�|��CT��                                    Bx�=�  �          @�(���p��G���
���\CS\��p��S�
��(��z�\CT��                                    Bx�=)�  �          @�z����
�J�H�Q���{CS�q���
�W������z�CU\)                                    Bx�=8B  �          @�z����N{�����  CS�����Y����ff�`��CUJ=                                    Bx�=F�  �          @�(���p��O\)����=qCT���p��Z=q��  �Yp�CUaH                                    Bx�=U�  �          @�(���p��L�Ϳ�z���33CS�R��p��W������d  CU!H                                    Bx�=d4  �          @��H�����aG���ff�c33CW\�����j=q��Q��,��CX)                                    Bx�=r�  �          @�=q�����c33��\)�H��CW{�����j�H��G���\CW�q                                    Bx�=��  �          @������H�j�H���R�2�\CW�{���H�q녿\(����CX�)                                    Bx�=�&  �          @љ�����vff��z��F�\CX������~{��G��ffCY�
                                    Bx�=��  �          @�=q��Q��e�����g�
CV8R��Q��n{���
�3�CWL�                                    Bx�=�r  �          @��H����X�ÿ�\)�dQ�CT�����a녿��
�3
=CU(�                                    Bx�=�  �          @�33��{�XQ��{�aCS����{�aG����\�0��CT��                                    Bx�=ʾ  T          @Ӆ��ff�Vff����e�CS����ff�`  ��ff�5��CT�R                                    Bx�=�d  �          @��H��  �Mp���
=�l��CR@ ��  �W
=��{�>�HCSff                                    Bx�=�
  �          @��H��ff�P�׿��H�qp�CR�f��ff�Z�H����B�\CT\                                    Bx�=��  �          @ҏ\���\�e�����J�HCU�����\�mp�����z�CV�)                                    Bx�>V  �          @�������^�R�����9CTǮ�����e�z�H�	�CU��                                    Bx�>�  �          @����z��aG����\�333CU(���z��hQ�n{�=qCU�                                    Bx�>"�  �          @����ff�W
=��z��F�RCS����ff�^�R��=q��
CT�\                                    Bx�>1H  �          @��H��  �W
=��{�>�\CSff��  �^{���
�Q�CTB�                                    Bx�>?�  �          @�z���33�Q녿�z��C�
CRT{��33�X�ÿ���33CS:�                                    Bx�>N�  �          @Ӆ��G��7���33�D  CNY���G��?\)�������COL�                                    Bx�>]:  �          @��
���H�3�
��{�>{CM�����H�:�H����  CN�
                                    Bx�>k�  �          @�33����3�
�����<Q�CM�\����;���=q�=qCN��                                    Bx�>z�  �          @�(���G��:�H����733CN�R��G��A녿���(�CO�{                                    Bx�>�,  �          @�33��z��e���\�1G�CU���z��l(��n{�p�CVn                                    Bx�>��  �          @љ���{�p�׿���8��CW�3��{�w��u�ffCX��                                    Bx�>�x  �          @�33��p��b�\����4  CU(���p��i���u�p�CU�                                    Bx�>�  �          @�=q��z��dzῑ�� (�CU�=��z��j=q�O\)��\CV33                                    Bx�>��  �          @�=q�����b�\��(��+
=CU5������hQ�c�
����CU�                                    Bx�>�j  �          @�=q��
=�\�Ϳ�
=�%�CTE��
=�c33�\(���Q�CT��                                    Bx�>�  �          @����G��S33�����((�CR��G��X�ÿc�
���\CSz�                                    Bx�>�  �          @��H���H�R�\��Q��&�\CR}q���H�XQ�c�
��Q�CS33                                    Bx�>�\  �          @������S33��Q��$(�CR33����X�ÿc�
��z�CR�f                                    Bx�?  �          @�����K����R�+\)CP������Q녿s33�33CQ�R                                    Bx�?�  �          @�����\)�I����p��*ffCP�{��\)�P  �p����RCQ��                                    Bx�?*N  �          @�{�����Fff���R�*�HCP&f�����L�Ϳu�(�CP��                                    Bx�?8�  �          @�{��G��G
=�����&{CP=q��G��Mp��n{���RCP��                                    Bx�?G�  T          @ָR����J=q��z��
=CP�\����P  �aG���Q�CQ=q                                    Bx�?V@  �          @�ff��G��Mp���ff���CP�q��G��R�\�E���33CQ�
                                    Bx�?d�  �          @�(������S�
��G����CR^������XQ�8Q��ȣ�CR�                                    Bx�?s�  �          @�p���p��U��{���CR����p��[��Q��ᙚCS.                                    Bx�?�2  �          @�{��\)�Q녿�����CQ�\��\)�W
=�E����
CRff                                    Bx�?��  �          @���Q��N{������RCQ.��Q��S33�J=q��  CQǮ                                    Bx�?�~  �          @�{��Q��L�Ϳ�� ��CQ\��Q��R�\�c�
���CQ��                                    Bx�?�$  �          @������O\)�Y����CQz�����S33����{CQ�                                    Bx�?��  �          @���
=�O\)��33��HCQ����
=�U��^�R��Q�CR.                                    Bx�?�p  �          @������S�
��z�� z�CRL�����Y���aG���=qCR�                                    Bx�?�  �          @�ff��
=�R�\��\)�G�CQ����
=�W��Tz����CR�                                    Bx�?�  �          @�p����R�Tz�n{� ��CR33���R�X�ÿ&ff��33CR��                                    Bx�?�b  �          @�����R�O\)��=q��CQ�����R�Tz�L����{CR(�                                    Bx�@  �          @�p����\�`�׿}p��	G�CT(����\�e��333����CT�                                    Bx�@�  �          @љ�����g��E��ָRCUǮ����j�H����=qCV&f                                    Bx�@#T  �          @�(���\)�,�ͿxQ��{CLL���\)�1G��=p���p�CL�)                                    Bx�@1�  �          @��
��\)�0�׿333����CL����\)�3�
�����G�CM.                                    Bx�@@�  �          @�����  �0�׿J=q��=qCL����  �4z�\)��33CM(�                                    Bx�@OF  �          @�����
=�5��E���CMff��
=�8�ÿ
=q��p�CM��                                    Bx�@]�  �          @�(����R�3�
�:�H�˅CM:����R�7
=��\���
CM��                                    Bx�@l�  �          @�������1녿8Q���{CL������5����H��\)CMJ=                                    Bx�@{8  �          @�(����R�3�
�0�����CM=q���R�7
=������CM�)                                    Bx�@��  �          @�������/\)�(����
=CLaH�����1녾�G��tz�CL��                                    Bx�@��  �          @���\�)���z�����CK���\�,(���p��K�CK޸                                    Bx�@�*  �          @����33�'
=�#�
����CK.��33�)����(��n�RCK�                                    Bx�@��  �          @�(���G��(�ÿ333����CK�)��G��,(����H��ffCK��                                    Bx�@�v  �          @�(������,(��(�����CL������.�R��ff�xQ�CLc�                                    Bx�@�  �          @��
��G��(Q�!G���  CK�\��G��+���(��l(�CK��                                    Bx�@��  �          @�������*=q��R��=qCK�f����,�;���`��CK�R                                    Bx�@�h  �          @���\�)�������
=CK���\�+���{�;�CK��                                    Bx�@�  �          @��
�\�#�
��\����CJ�=�\�%���R�*�HCK
=                                    Bx�A�  �          @��
��33�!G�������CJn��33�#33�����5�CJ��                                    Bx�AZ  �          @�33��=q�"�\����{CJ�3��=q�$zᾏ\)�{CJ�                                    Bx�A+   �          @�33�Å�(�����ffCI���Å�{��z��#�
CI�R                                    Bx�A9�  �          @Ӆ���
��H�   ����CI�����
��;��R�+�CI��                                    Bx�AHL  �          @Ӆ�Å�{�   ��33CI�R�Å�   ���R�,��CJ8R                                    Bx�AV�  �          @����=q��;���g�CJ��=q��R�k���p�CJ8R                                    Bx�Ae�  �          @У���  � �׾�����CJ�3��  �"�\��=q�
=CJ�                                    Bx�At>  �          @љ���  �$z�   ��p�CK+���  �&ff���R�-p�CKk�                                    Bx�A��  �          @������(Q���H��=qCK������*=q�����%�CK�f                                    Bx�A��  �          @љ�����'��   ��p�CK�f����)�����R�,(�CK�f                                    Bx�A�0  �          @����  �'���G��xQ�CK�{��  �(�þ�  �
�HCK�=                                    Bx�A��  �          @��H��  �*=q����{CK����  �,�;�p��L��CL=q                                    Bx�A�|  �          @�z���  �2�\�����
CL����  �4zᾊ=q�z�CM.                                    Bx�A�"  �          @Ӆ�����*�H�   ��33CK�������,�;����(Q�CL�                                    Bx�A��  �          @�(���G��,(������33CL  ��G��.{��=q��CL8R                                    Bx�A�n  �          @�p����H�+���(��o\)CK����H�-p��u��CK��                                    Bx�A�  �          @�ff����'�����b�\CK�����(�þaG����CK5�                                    Bx�B�  �          @�ff�����(Q쾨���5�CK������(�þ\)��Q�CKB�                                    Bx�B`  
�          @���(��(�þ�{�:=qCKE��(��)�������G�CKh�                                    Bx�B$  �          @�ff��p��$zᾮ{�9��CJ�f��p��%������CJ�=                                    Bx�B2�  �          @ָR��
=� �׾��
�-p�CI����
=�!녾���33CJ)                                    Bx�BAR  �          @����!G���z���RCJ0����!녽�G��k�CJL�                                    Bx�BO�  �          @�ff��ff�!G���\)�Q�CJ!H��ff�"�\���ͿTz�CJ=q                                    Bx�B^�  �          @�{��{�!녾�\)��CJE��{�#33���ͿO\)CJ^�                                    Bx�BmD  �          @�{����%�����$z�CJ�����&ff��G��z�HCJ��                                    Bx�B{�  �          @�ff���%������#�
CJ�H���%���}p�CJ�                                     Bx�B��  �          @�ff��\)�(���\)�Q�CIc���\)��ͽ��Ϳc�
CI�                                     Bx�B�6  �          @�ff�ƸR�   ��  �
�HCI��ƸR� �׽��
�&ffCJ�                                    Bx�B��  ;          @�p���Q���׾�{�<(�CG�)��Q��녾8Q�ǮCH�                                    Bx�B��  �          @�{�������
=�c�
CG�����p������RCGJ=                                    Bx�B�(  �          @�
=��=q�  ���
�,��CG����=q�G��#�
����CG�                                     Bx�B��  �          @�{�ə��\)�����$z�CG���ə��  �\)��p�CG��                                    Bx�B�t  �          @���G���R��\)��HCGz���G��\)����=qCG��                                    Bx�B�  �          @�ff�ə��G��u�z�CG���ə��녽��
�5CG��                                    Bx�B��  �          @ָR��G��zᾣ�
�.�RCHJ=��G���#�
���CHk�                                    Bx�Cf  �          @ָR�ə��녾�{�7
=CG��ə��33�8Q��G�CH\                                    Bx�C  �          @ָR�ȣ��Q쾳33�AG�CH��ȣ�����B�\��{CH�                                    Bx�C+�  �          @�{�ȣ��녾�p��J=qCG�3�ȣ��33�W
=��ffCH�                                    Bx�C:X  �          @���\)������`��CH�{��\)��þ�  �
=CI�                                    Bx�CH�  �          @�����  ��׾�=q��
CG���  �녽�G��xQ�CH�                                    Bx�CW�  �          @�ff������
�u�CH+������zὸQ�:�HCHB�                                    Bx�CfJ  �          @�{������\�aG���
=CH�����33��\)�z�CH�                                    Bx�Ct�  �          @����\)��u�z�CH����\)�
=���
�0��CH��                                    Bx�C��  �          @�p���\)�Q�.{���HCH�H��\)�Q�#�
��\)CH�                                    Bx�C�<  �          @�ff�Ǯ�=q<#�
=���CI#��Ǯ�=q>.{?�  CI
                                    Bx�C��  �          @�ff��  ���>L��?��HCH�f��  ��>�Q�@ECH�                                     Bx�C��  �          @�{��
=�{>\)?�p�CI�f��
=���>��R@*=qCI��                                    Bx�C�.  �          @�ff��ff�!�=#�
>�p�CJ+���ff�!G�>W
=?�=qCJ�                                    Bx�C��  �          @��������"�\������CJn�����!�>\)?�CJh�                                    Bx�C�z  �          @�{���#�
<#�
=L��CJu����#33>8Q�?\CJh�                                    Bx�C�   �          @�Q���Q��!�=�G�?h��CI�3��Q�� ��>�\)@ffCIٚ                                    Bx�C��  �          @׮�ȣ���R=���?Y��CI���ȣ��{>�=q@��CI�                                     Bx�Dl  �          @�  ��=q�ff>k�?�p�CHs3��=q��>Ǯ@S�
CHG�                                    Bx�D  �          @أ���z��{>\@N{CG33��z��(�?�@�
=CF��                                    Bx�D$�  �          @������
�\)>�=q@  CGff���
�{>��@aG�CG8R                                    Bx�D3^  �          @ٙ��ə��녿�z��p�CG�f�ə��ff��G��Q�CHz�                                    Bx�DB  �          @��H��33�녿�{��CG����33��s33� Q�CHB�                                    Bx�DP�  �          @ٙ���=q�G����
��CG�R��=q���aG���CH:�                                    Bx�D_P  �          @ٙ���(��{�J=q��p�CG.��(��G��#�
��p�CG�\                                    Bx�Dm�  �          @�33��{�\)��R��ffCG.��{�녾���|(�CGu�                                    Bx�D|�  T          @����z��G��
=��ffCG����z��33��G��k�CG��                                    Bx�D�B  �          @׮�ʏ\�  �
=����CG���ʏ\��\��G��p  CG��                                    Bx�D��  �          @�G���(����E���Q�CF�H��(���R��R����CG=q                                    Bx�D��  �          @ڏ\��p���R�(�����CG(���p��G���\���CGxR                                    Bx�D�4  �          @�33��p��33�!G���G�CG�3��p�������Q�CG��                                    Bx�D��  �          @�33�θR��Ϳ\)��{CF���θR��R��
=�^{CG�                                    Bx�DԀ  �          @�=q��z���׿+����
CG}q��z��33�����CG��                                    Bx�D�&  T          @׮��33��Ϳ���z�CG���33��R��(��i��CGc�                                    Bx�D��  �          @׮���H�p��z����CG.���H�\)��(��j�HCGp�                                    Bx�E r  T          @�ff�ə��p��
=q���CGO\�ə��\)�����Z=qCG�\                                    Bx�E  �          @�{�ə��{�Ǯ�XQ�CGk��ə��\)��  �Q�CG�
                                    Bx�E�  �          @�
=���
��
?�\@��
CE�3���
��?#�
@�Q�CE��                                    Bx�E,d  �          @����θR��\>�\)@�CE���θR�G�>��@\��CEY�                                    Bx�E;
  �          @�
=�����  ?:�H@ȣ�CG�q�������?aG�@�Q�CGO\                                    Bx�EI�  �          @׮���H�{?��@���CGJ=���H��?0��@��
CF�R                                    Bx�EXV  �          @׮�����?��@�z�CG޸����\)?0��@���CG�=                                    Bx�Ef�  �          @أ����H��?�@��
CH8R���H�33?+�@���CG��                                    Bx�Eu�  �          @�=q���
���?   @�CH�=���
�ff?&ff@�
=CH@                                     Bx�E�H  �          @������H���>\@L(�CH�H���H�
=?�@��CHff                                    Bx�E��  �          @׮�ȣ��p�>���@1G�CIn�ȣ���>�@��
CI8R                                    Bx�E��  �          @�\)�Ǯ�{>�33@=p�CI���Ǯ�(�?�\@�=qCIY�                                    Bx�E�:  �          @�
=��
=�   >Ǯ@W�CIٚ��
=�{?\)@��CI�)                                    Bx�E��  �          @�Q���\)�'
=>�\)@CJ�q��\)�%>�G�@qG�CJ�\                                    Bx�E͆  �          @أ��Ǯ�'�>k�?��HCJ�Ǯ�&ff>���@X��CJ��                                    Bx�E�,  �          @�����Q��%�>aG�?���CJn��Q��#�
>\@O\)CJG�                                    Bx�E��  �          @׮�Ǯ�!G�>��?���CJ�Ǯ� ��>��
@,��CI�f                                    Bx�E�x  �          @ָR��{�%<�>�\)CJ� ��{�%>L��?�(�CJ��                                    Bx�F  �          @�=q�ȣ��+���\)�\)CK)�ȣ��+�=�G�?fffCK�                                    Bx�F�  �          @�p����H�0  ��Q�=p�CKs3���H�0  =�Q�?=p�CKs3                                    Bx�F%j  �          @�ff��p��*�H�B�\���
CJ�\��p��*�H�#�
��G�CJ�)                                    Bx�F4  �          @�{����)���W
=��p�CJxR����*=q�#�
���RCJ��                                    Bx�FB�  �          @�
=��ff�(Q쾊=q���CJ+���ff�(�ý��ͿL��CJB�                                    Bx�FQ\  �          @�
=��z��2�\��G��aG�CK�
��z��2�\=�\)?(�CK�
                                    Bx�F`  �          @�{���
�/\)�L�ͿУ�CKE���
�0  ���
�#�
CKT{                                    Bx�Fn�  T          @����z��'
=��G��k�CJ=q��z��'�=u>��HCJB�                                    Bx�F}N  �          @�ff�����,(������,(�CJ�=�����,�;�����RCJ��                                    Bx�F��  �          @�ff��(��.{�����(�CK���(��.�R���}p�CK&f                                    Bx�F��  �          @�p����
�,(������\)CJ�����
�,�;����CJ�q                                    Bx�F�@  �          @�{����(Q��G��fffCJT{����*=q�����CJ��                                    Bx�F��  �          @�Q���\)�'
=�z���\)CI����\)�(�þ���U�CJ(�                                    Bx�Fƌ  �          @�\��G��*�H��
=�X��CJ=q��G��,�;u��p�CJff                                    Bx�F�2  �          @�=q�����,�;�33�5CJu������.{�.{��z�CJ�{                                    Bx�F��  �          @�33��G��/\)�u��
=CJ� ��G��0  �u���HCJ��                                    Bx�F�~  m          @���G��0  ���
�#�
CJ�\��G��1G��\)��{CJ�                                    Bx�G$  	�          @���θR�333��ff�l(�CKk��θR�4zᾊ=q���CK��                                    Bx�G�  
�          @�=q��\)�0�׾��{�CK���\)�2�\�����p�CKJ=                                    Bx�Gp  T          @���G��0�׾��H�z�HCJ����G��2�\���R�p�CK{                                    Bx�G-  T          @��
�љ��0  ���w
=CJ��љ��1G��������CJ��                                    Bx�G;�  �          @���Q��3�
��\��z�CKQ���Q��5�����)��CK��                                    Bx�GJb  �          @��H��
=�5��+�����CK����
=�7����x��CK�                                    Bx�GY  �          @���
=�7
=�8Q�����CKٚ��
=�9������  CL&f                                    Bx�Gg�  "          @��
�θR�:=q�5��
=CLB��θR�<�Ϳ���p�CL��                                    Bx�GvT  �          @���p��>{�0������CLǮ��p��@  ���H�~{CM�                                    Bx�G��  �          @�33��
=�7��
=q���\CK���
=�9����33�333CL#�                                    Bx�G��  �          @��H��(��   ��p��>{CH�H��(�� �׾L�ͿУ�CHǮ                                    Bx�G�F  
�          @�33�����{��=q�
�HCHW
������R��G��Y��CHn                                    Bx�G��  
�          @��
����!G��\)��\)CH������!G�<�>aG�CH�q                                    Bx�G��  "          @�G���=q�"�\�#�
��G�CI!H��=q�!�>��?�G�CI�                                    Bx�G�8  �          @����ff�1G��#�
����CK=q��ff�0��>��?���CK8R                                    Bx�G��  
�          @�Q���ff�/\)=�G�?aG�CK  ��ff�.{>�z�@
=CJ�f                                    Bx�G�  �          @�����=q�   >u?��HCH�{��=q��R>��@S33CH��                                    Bx�G�*  �          @���ָR��>W
=?޸RCF��ָR�
=q>�Q�@:�HCE�)                                    Bx�H�  �          @ᙚ��ff���>�=q@p�CF8R��ff��>�
=@Y��CF
=                                    Bx�Hv  �          @�  �����Q�?\)@��
CE�����?333@���CEn                                    Bx�H&  "          @�ff��=q��?aG�@�G�CE����=q� ��?��\A�HCE�                                    Bx�H4�  �          @�ff���
���R?@  @��CD�����
��
=?aG�@�  CDB�                                    Bx�HCh  �          @���z��Q�>��@Z�HCDT{��z��z�?
=q@�  CD�                                    Bx�HR  �          @������
��33>�G�@i��CD  ���
��\)?�@��RCC��                                    Bx�H`�  �          @����Ӆ����>��@[�CDp��Ӆ��?
=q@�Q�CD.                                    Bx�HoZ  
�          @����
���R>�G�@i��CD���
���H?�@�Q�CD}q                                    Bx�H~   �          @��������?(��@��RCE8R��녿�p�?J=q@��HCD�{                                    Bx�H��  ;          @�(��љ����H?G�@�Q�CD�H�љ���33?h��@�CD+�                                    Bx�H�L  ;          @�����33��
=?
=@���CDE��33���?8Q�@�\)CC��                                    Bx�H��  �          @�z���(���\?��@��CB���(���p�?+�@���CB��                                    Bx�H��  m          @�z���z�޸R?(��@�
=CB���z��Q�?E�@�{CBJ=                                    Bx�H�>  
�          @�����z���
?
=@�p�CC���z�޸R?5@�p�CB��                                    Bx�H��  "          @��
��33��?�@�ffCCT{��33��\?(��@�\)CC                                      Bx�H�  �          @ڏ\�ҏ\���H?.{@�ffCB�\�ҏ\��z�?J=q@�p�CB&f                                    Bx�H�0  "          @ٙ���녿�
=?.{@�
=CBY���녿У�?J=q@�CA�                                    Bx�I�  T          @�33�Ӆ��?&ff@�\)CB.�Ӆ��\)?E�@�{CAǮ                                    Bx�I|  �          @��
�ҏ\��p��\)��
=CD� �ҏ\��p�    ��CD�=                                    Bx�I"  �          @ۅ�Ӆ��    �#�
CC�\�Ӆ��>�?�ffCC�                                    Bx�I-�  �          @��
���Ϳ�Q�>B�\?���CBJ=���Ϳ�
=>���@"�\CB&f                                    Bx�I<n  
�          @��
��(����>�=q@p�CC���(���\>Ǯ@P  CB�                                    Bx�IK  "          @��
��(����
>u@G�CC���(���G�>�p�@C33CB�H                                    Bx�IY�  �          @�33��z��
=>�?���CB@ ��z��>u@33CB#�                                    Bx�Ih`  �          @ڏ\�Ӆ���H=��
?(��CBz��Ӆ�ٙ�>L��?�z�CBff                                    Bx�Iw  	�          @ٙ���33�У�=L��>���CA����33��\)>#�
?�\)CA�{                                    Bx�I��  "          @ٙ��ҏ\��
==���?Q�CBO\�ҏ\��>W
=?�CB8R                                    Bx�I�R  �          @�Q����ÿ�p���z����CB�����ÿ޸R�.{��z�CB�                                    Bx�I��  T          @ָR�θR���
����  CCh��θR��ff�\)��CC��                                    Bx�I��  �          @�  ��Q��  �\�N{CC���Q���
��  �	��CC@                                     Bx�I�D  ;          @�Q��У׿�(����y��CB�=�У׿�  ��{�6ffCC                                    Bx�I��  ;          @�����Q��{��
=�dz�CC���Q��׾�z��(�CD�                                    Bx�Iݐ  
�          @�Q���\)��{��33�<(�CD���\)��׾W
=��ffCD+�                                    Bx�I�6  
�          @�  ��\)��\)��{�7�CD)��\)��녾L�Ϳ�p�CDE                                    Bx�I��  T          @�Q��Ϯ���;����%�CC�H�Ϯ��{�.{��Q�CD                                    Bx�J	�  T          @ٙ��љ���=q�u���HCC�f�љ���=q=�\)?�RCC��                                    Bx�J(  �          @�G���녿�  �\)��CB�f��녿�  �#�
����CB�3                                    Bx�J&�  T          @ڏ\�Ӆ��p��B�\��=qCB�f�Ӆ�޸R�u��CB��                                    Bx�J5t  �          @�(�����޸R���
��CB�H����޸R=�G�?n{CB�)                                    Bx�JD  <          @�z������  =�Q�?8Q�CB�R����޸R>W
=?��
CB�H                                    